import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from modules.merge_layer import MergeLayer

class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, temperature,attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)
        

    def forward(self, q, k, v,seq_vnE, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = attn + seq_vnE
        
        
        attn = self.softmax(attn) 
        attn = self.dropout(attn) 
        output = torch.bmm(attn, v)
        
        return output, attn



class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v,dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v,seq_vnE, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) 
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) 

        mask = mask.repeat(n_head, 1, 1) 
        output, attn = self.attention(q, k, v,seq_vnE, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) 

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        
        return output, attn
 
class AttnModel(torch.nn.Module):

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):

        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        self.n_head = n_head
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head = n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out
                                             )
            self.logger.info('Using scaled prod attention')
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e,seq_vnE, mask):

        seq_vnE = seq_vnE.repeat(self.n_head,1,1).view([src.size(0) * self.n_head,1,-1])
        src_ext = torch.unsqueeze(src, dim=1) 
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) 
        k = torch.cat([seq, seq_e, seq_t], dim=2) 
        
        mask = torch.unsqueeze(mask, dim=2) 
        mask = mask.permute([0, 2, 1]) 
        
        
        
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask,seq_vnE = seq_vnE) 
        output = output.squeeze()
        attn = attn.squeeze() 

        output = self.merger(output, src)
        return output, attn

