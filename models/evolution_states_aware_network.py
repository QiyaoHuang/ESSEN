
import logging

import numpy as np
import torch
import modules
import modules.aggreation as aggreation
import logging
logging.getLogger().setLevel(logging.INFO)

class ESSEN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat,num_layers, n_head,
                 max_expand_subgraph_size,num_expert,
                 attn_mode='prod', use_time='time',
                 drop_out=0.1,  device = 'cpu'):
        super(ESSEN, self).__init__()

        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.device = device
        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim 
        self.model_dim = self.feat_dim
        self.use_time = use_time
        self.max_expand_subgraph_size = max_expand_subgraph_size

        self.output_dim = self.feat_dim 

        self.logger.info('Aggregation uses attention model')
        self.attn_model_list = torch.nn.ModuleList([aggreation.AttnModel(self.feat_dim, 
                                                            self.feat_dim, 
                                                            self.feat_dim,
                                                            attn_mode=attn_mode, 
                                                            n_head=n_head, 
                                                            drop_out=drop_out) for _ in range(num_layers)])
        

        self.time_encoder = modules.TimeEncode(expand_dim=self.n_feat_th.shape[1])
        self.entropy_encoder = modules.TimeEncode(expand_dim=self.n_feat_th.shape[1])
        self.moe_layer = modules.SparseMoE(input_gate_size=3,input_embed_size = self.output_dim*4,num_experts=num_expert,
                                            output_expert_size=self.output_dim)
    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder   

    def encoder(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors):

        change_src_embed_with_target,_ = self.forward(src_idx_l, cut_time_l, self.num_layers, num_neighbors,target_idx_l,'change')


        change_target,_ = self.forward(target_idx_l, cut_time_l, self.num_layers, num_neighbors,src_idx_l,'change')

        return change_src_embed_with_target,change_target

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors):
        src_embed,_ = self.forward(src_idx_l, cut_time_l, self.num_layers, num_neighbors,[],'orignal')
        
        target_embed,_ = self.forward(target_idx_l, cut_time_l, self.num_layers, num_neighbors,[],'orignal')

        background_embed,_ = self.forward(background_idx_l, cut_time_l, self.num_layers, num_neighbors,[],'orignal')
        
        change_src_embed_with_target,change_target  = self.encoder(src_idx_l, target_idx_l, cut_time_l, num_neighbors)
        change_src_embed_with_background,change_background  = self.encoder(src_idx_l, background_idx_l, cut_time_l, num_neighbors)


        pos_energy_np = np.zeros((len(src_idx_l),3))
        neg_energy_np = np.zeros((len(src_idx_l),3))


        for i in range(len(src_idx_l)):
            src_idx = src_idx_l[i]
            tail_idx =  target_idx_l[i]
            cut_time = cut_time_l[i]
            background_idx = background_idx_l[i]
            pos_energy_np[i,:] = self.ngh_finder.computation_entropy_tempurature(src_idx,tail_idx,cut_time)
            neg_energy_np[i,:] = self.ngh_finder.computation_entropy_tempurature(src_idx,background_idx,cut_time)
        pos_energy_th  = torch.from_numpy(np.array(pos_energy_np)).float().to(self.device) 
        neg_energy_th  =  torch.from_numpy(np.array(neg_energy_np)).float().to(self.device)  
        pos_score = self.moe_layer(pos_energy_th,torch.cat([src_embed,target_embed, change_src_embed_with_target - src_embed , change_target - target_embed],dim = 1)).squeeze(dim=-1)  
        neg_score = self.moe_layer(neg_energy_th,torch.cat([src_embed,background_embed,  change_src_embed_with_background-src_embed,change_background - background_embed],dim = 1)).squeeze(dim=-1) 
            
        return pos_score.sigmoid(), neg_score.sigmoid()

    def forward(self, node_idx_l, cut_time_l, curr_layers, num_neighbors,edge_another_node_idx_l, node_type):   
            
        final_emb,weight = self.hetero_tem_conv(node_idx_l = node_idx_l, cut_time_l = cut_time_l, curr_layers = curr_layers,
                                                num_neighbors = num_neighbors,final_node_idx_l = node_idx_l, final_cut_time_l = cut_time_l,edge_another_node_idx_l = edge_another_node_idx_l,
                                                node_type = node_type)
        return final_emb,weight

    def hetero_tem_conv(self, node_idx_l, cut_time_l, curr_layers, num_neighbors,final_node_idx_l,final_cut_time_l,edge_another_node_idx_l, node_type):
        assert(curr_layers >= 0)

        batch_size = len(node_idx_l)
        
        src_node_batch_th = torch.from_numpy(node_idx_l).long().to(self.device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(self.device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)
        
        if curr_layers == 0:
            return src_node_feat,0
        else:
            src_node_conv_feat,_f = self.hetero_tem_conv(node_idx_l = node_idx_l, 
                                           cut_time_l = cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors,final_node_idx_l = final_node_idx_l,final_cut_time_l = final_cut_time_l,
                                            edge_another_node_idx_l = edge_another_node_idx_l,
                                            node_type = node_type)

            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch,src_ngh_vnE_batch =\
            self.ngh_finder.get_heterogeneous_neighbor( final_node_idx_l = final_node_idx_l,
                                                        final_cut_time_l = final_cut_time_l,
                                                        node_idx_l = node_idx_l,
                                                        cut_time_l = cut_time_l,
                                                        num_neighbors=num_neighbors,
                                                        max_expand_subgraph_size = self.max_expand_subgraph_size,
                                                        edge_another_node_idx_l = edge_another_node_idx_l,
                                                        node_type = node_type)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(self.device)
            src_ngh_vnE_batch= torch.from_numpy(src_ngh_vnE_batch).float().to(self.device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)
            
            
            
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() 
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() 

            src_ngh_node_conv_feat,_s = self.hetero_tem_conv(node_idx_l = src_ngh_node_batch_flat, 
                                                   cut_time_l = src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors,
                                                   final_node_idx_l = final_node_idx_l,
                                                   final_cut_time_l = final_cut_time_l,
                                                    edge_another_node_idx_l = edge_another_node_idx_l,
                                                    node_type = node_type)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)


            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src = src_node_conv_feat, 
                                   src_t = src_node_t_embed,
                                   seq = src_ngh_feat,
                                   seq_t = src_ngh_t_embed, 
                                   seq_e = src_ngn_edge_feat, 
                                   seq_vnE = src_ngh_vnE_batch,
                                   mask = mask
                                   )
            return local, weight
