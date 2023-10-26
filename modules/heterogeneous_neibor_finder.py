from re import L
import numpy as np
import random
import collections
import copy
class HeteroNeighborFinder:
    def __init__(self, adj_list ,quadruple):
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)  
        self.quadruple_src,self.quadruple_dst,self.quadruple_e,self.quadruple_time = zip(*quadruple)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        self.off_set_l = off_set_l
        self.node_num = len(set(node_idx_l))
        self.edge_num = len(self.edge_idx_l)
        self.neighbor_dict = {}

        self.degree_hash_dict = collections.defaultdict(lambda:[])
        self.v_count = 0
        self.e_count = 0
        self.current_time = -1
        self.current_idx = -1
        self.entropy_tempurature_dict = {}
        self.current_graph_count_dict = {}
    def init_off_set(self, adj_list):

            n_idx_l = []
            n_ts_l = []
            e_idx_l = []
            off_set_l = [0]
            length = len(adj_list) 

            for i in range(length):
                curr = adj_list[i]
                curr = sorted(curr, key=lambda x: x[1],reverse = False)
                n_idx_l.extend([x[0] for x in curr])
                e_idx_l.extend([x[1] for x in curr])
                n_ts_l.extend([x[2] for x in curr])
            
                
                off_set_l.append(len(n_idx_l))
            n_idx_l = np.array(n_idx_l)
            n_ts_l = np.array(n_ts_l)
            e_idx_l = np.array(e_idx_l)
            off_set_l = np.array(off_set_l)

            assert(len(n_idx_l) == len(n_ts_l))
            assert(off_set_l[-1] == len(n_ts_l))
            
            return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before_degree(self, src_idx, cut_time):

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        left = -1
        right = len(neighbors_idx)
        
        while left + 1 != right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t <= cut_time:
                left = mid
            else:
                right = mid

        return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right],right

    
    def current_graph_count(self,cut_time):
        if cut_time in self.current_graph_count_dict:
            return self.current_graph_count_dict[cut_time]
        if self.current_time < cut_time:
            left,right = self.current_idx,len(self.quadruple_time)
        else:
            left,right = -1,len(self.quadruple_time)
        while left + 1 != right:
            mid = (left + right) // 2
            curr_t = self.quadruple_time[mid]
            if curr_t <= cut_time:
                left = mid
            else:
                right = mid   
        self.current_time = cut_time
        self.current_idx = left
        self.e_count = right
        self.v_count = len(set(self.quadruple_src[:right] + self.quadruple_dst[:right]))
        self.current_graph_count_dict[cut_time] = (self.v_count,self.e_count)
        return self.v_count,self.e_count

    def get_von_Neumann_entropy_each(self,du,dv,cut_time):
        if du == 0 :
            du = 1
        if dv == 0 :
            dv = 1
        v_count,e_count = self.current_graph_count(cut_time)
        entropy = ((1 -(1/v_count)-(1/(e_count**2 * du*dv ))) / e_count)
        return entropy      
    
    def get_heterogeneous_neighbor_find_before(self, src_idx, cut_time):
        ngh_idx, ngh_eidx, ngh_ts,du  = self.find_before_degree(src_idx, cut_time)

        length_ngh = len(ngh_idx)
        

        length = length_ngh
        ngh_idx = ngh_idx.tolist()
        neighor_list = []

        
        if length == 0 : 
            return [],0
        for i in range(length):
            
            v_idx = ngh_idx[i]
            
            if v_idx not in self.already_ex_node_list:
                count = ngh_idx.count(v_idx)
                time = ngh_ts[i]
                _,_,_,dv= self.find_before_degree(v_idx,time)

                entropy = self.get_von_Neumann_entropy_each(du = du,dv = dv,cut_time = cut_time)            
       
                if src_idx <= ngh_idx[i]:
                    neighor_list.append((src_idx,ngh_idx[i],ngh_eidx[i],time,count,entropy)) 
                else:
                    neighor_list.append((ngh_idx[i],src_idx,ngh_eidx[i],time,count,entropy))

        return neighor_list,ngh_idx 

    def set_candidate_edge_set_list(self,src_idx_l,cut_time_l,max_expand_subgraph_size):
        hop = 2
        f_neighor_list_all = []
        for node_index in range(len(src_idx_l)):                
            
            expand_node_idx = src_idx_l[node_index]
            query_time =  cut_time_l[node_index]
            
            f_neighor_list = []
            
            self.already_ex_node_list = []
            expand_node_list = [(expand_node_idx,query_time)]
            
            
            
            cut_time = query_time
            for i in range(hop):  
                
                neibor_hop_list = []
                while len(expand_node_list) != 0 and len(f_neighor_list) < max_expand_subgraph_size: 
                    src_node = expand_node_list.pop(0)
                    src_idx = src_node[0]
                    
                    if src_idx not in self.already_ex_node_list:
                        self.already_ex_node_list.append(src_idx)
                        
                        
                        
                        neighor_list,ngh_idx = self.get_heterogeneous_neighbor_find_before(src_idx, cut_time)
                        
                        f_neighor_list = f_neighor_list + neighor_list
                        if neighor_list != []:
                            max_cut_time = max(neighor_list,key=lambda x:x[-3])[-3]
                            neibor_hop_list = neibor_hop_list + [(x,max_cut_time) for x in ngh_idx]

                neibor_hop_list = list(set(neibor_hop_list))
                expand_node_list = neibor_hop_list

            if len(f_neighor_list) == 0:
                f_neighor_list_all.append([ (src_idx_l[node_index], src_idx_l[node_index],0,0.0,0,0)])
            else:
                f_neighor_list_all.append(f_neighor_list)
           
        return f_neighor_list_all


    def list2dict(self,src_idx_l,cut_time_l,f_neighor_list_all):
        for i in range(len(f_neighor_list_all)):
            f_neighor_list = f_neighor_list_all[i]
            src_idx = src_idx_l[i]
            cut_time = cut_time_l[i]
            if src_idx not in self.neighbor_dict:
                self.neighbor_dict[src_idx] = {}
            if cut_time not in self.neighbor_dict[src_idx]:
                self.neighbor_dict[src_idx][cut_time] = {}    
                        
            for new_edge in f_neighor_list:
                head_node = new_edge[0]
                tail_node = new_edge[1]

                if head_node not in self.neighbor_dict[src_idx][cut_time]:
                    self.neighbor_dict[src_idx][cut_time][head_node] = []
                
                self.neighbor_dict[src_idx][cut_time][head_node].append(new_edge[1:])

                if tail_node not in self.neighbor_dict[src_idx][cut_time]:
                    self.neighbor_dict[src_idx][cut_time][tail_node] = []
                
                self.neighbor_dict[src_idx][cut_time][tail_node].append((new_edge[0:1]+new_edge[2:]))
    
    def set_heterogeneous_neighbor(self,src_idx_l,cut_time_l,max_expand_subgraph_size):
        candidate_edge_set_list = self.set_candidate_edge_set_list(src_idx_l,cut_time_l,max_expand_subgraph_size)
        self.list2dict(src_idx_l,cut_time_l,candidate_edge_set_list)

    def find_tree(self,i,num_neighbors):
        idx = 0
        while i >= (idx+1) * num_neighbors:
            idx += 1
        return idx
    
    def get_heterogeneous_neighbor(self,final_node_idx_l,final_cut_time_l,node_idx_l,cut_time_l,num_neighbors,max_expand_subgraph_size,edge_another_node_idx_l, node_type):

        if node_type == 'orignal' or len(final_node_idx_l) != len(node_idx_l) :
            return self.get_heterogeneous_neighbor_orignal(final_node_idx_l,final_cut_time_l,node_idx_l,cut_time_l,num_neighbors,max_expand_subgraph_size)
        else:
            length = len(node_idx_l)
            out_ngh_node_batch = np.zeros((length, num_neighbors)).astype(np.int32)
            out_ngh_t_batch = np.zeros((length, num_neighbors)).astype(np.float32)
            out_ngh_eidx_batch = np.zeros((length, num_neighbors)).astype(np.int32)
            out_ngh_vnE_batch = np.zeros((length, num_neighbors)).astype(np.float64)   
            for i,node in enumerate(node_idx_l) :
                if node == 0:
                    continue
                idx = self.find_tree(i,num_neighbors)
                final_src_idx = final_node_idx_l[idx] 
                final_cut_time = final_cut_time_l[idx]
                edge_another_node_idx = edge_another_node_idx_l[idx]
                self.set_heterogeneous_neighbor([final_src_idx],[final_cut_time],max_expand_subgraph_size)       
                self.set_heterogeneous_neighbor([edge_another_node_idx],[final_cut_time],max_expand_subgraph_size)
            
                neibor = self.neighbor_dict[final_src_idx][final_cut_time][final_src_idx] 
                neibor_two = self.neighbor_dict[edge_another_node_idx][final_cut_time][edge_another_node_idx]                 
                neibor_node  = list(set([x[0] for x in neibor ]))
                neibor_two_node = list(set([x[0] for x in neibor_two ]))

                du = len(neibor_node)+1
                dv = len(neibor_two_node)+1
                fake_entropy = self.get_von_Neumann_entropy_each(du = du,dv = dv,cut_time=final_cut_time)
                
                neibor.append((edge_another_node_idx,9999,final_cut_time,1,fake_entropy))
                neibor.sort(key = lambda x: x[2], reverse= True)
                node_l = [ ]
                eidx_l = []
                t_l = []
                entropy_l = []
                for x in neibor:
                    node_l.append(x[0])  
                    eidx_l.append(x[1])  
                    t_l.append(x[2])  
                    entropy_l.append(x[4])  
                length_neibor = len(node_l[:num_neighbors])

                if length_neibor > 0:
                    out_ngh_node_batch[i, num_neighbors - length_neibor:] = node_l[:num_neighbors]
                    out_ngh_eidx_batch[i,  num_neighbors - length_neibor:] = eidx_l[:num_neighbors]         
                    out_ngh_t_batch[i, num_neighbors - length_neibor:] = t_l[:num_neighbors]
                    out_ngh_vnE_batch[i,  num_neighbors - length_neibor:] = entropy_l[:num_neighbors]  

            return  out_ngh_node_batch,out_ngh_eidx_batch,out_ngh_t_batch,out_ngh_vnE_batch

    def get_heterogeneous_neighbor_orignal(self,final_src_idx_l,final_cut_time_l,src_idx_l,cut_time_l,num_neighbors,max_expand_subgraph_size):

        length = len(src_idx_l)
        out_ngh_node_batch = np.zeros((length, num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((length, num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((length, num_neighbors)).astype(np.int32)
        out_ngh_vnE_batch = np.zeros((length, num_neighbors)).astype(np.float64)


        for i,node in enumerate(src_idx_l) :
            if node == 0:
                continue       
            if len(final_src_idx_l) == len(src_idx_l):
                idx = i
            else:
                idx = self.find_tree(i,num_neighbors)
            final_src_idx = final_src_idx_l[idx] 
            final_cut_time = final_cut_time_l[idx]
            if final_src_idx in self.neighbor_dict and final_cut_time in self.neighbor_dict[final_src_idx]:         
                if node in self.neighbor_dict[final_src_idx][final_cut_time]:
                    neibor = self.neighbor_dict[final_src_idx][final_cut_time][node] 
            else:            
                self.set_heterogeneous_neighbor([final_src_idx],[final_cut_time],max_expand_subgraph_size)
                if node in self.neighbor_dict[final_src_idx][final_cut_time]:
                    neibor = self.neighbor_dict[final_src_idx][final_cut_time][node] 
                node_l = [ ]
                eidx_l = []
                t_l = []
                entropy_l = []
                for x in neibor:
                    node_l.append(x[0])  
                    eidx_l.append(x[1])  
                    t_l.append(x[2])  
                    entropy_l.append(x[4])  
                length_neibor = len(node_l[:num_neighbors])

                if length_neibor > 0:
                    out_ngh_node_batch[i, num_neighbors - length_neibor:] = node_l[:num_neighbors]
                    out_ngh_eidx_batch[i,  num_neighbors - length_neibor:] = eidx_l[:num_neighbors]         
                    out_ngh_t_batch[i, num_neighbors - length_neibor:] = t_l[:num_neighbors]
                    out_ngh_vnE_batch[i,  num_neighbors - length_neibor:] = entropy_l[:num_neighbors]  
                    

        return  out_ngh_node_batch,out_ngh_eidx_batch,out_ngh_t_batch,out_ngh_vnE_batch

    def hash_degree(self,v,t):
        if (v,t) not in self.degree_hash_dict:
            idx_vneighbor,_,_,d_v = self.find_before_degree(v,t)
            self.degree_hash_dict[(v,t)].append(d_v)
            self.degree_hash_dict[(v,t)].append(dict(collections.Counter(idx_vneighbor)))
            return self.degree_hash_dict[(v,t)][0],self.degree_hash_dict[(v,t)][1]
        else:
            return self.degree_hash_dict[(v,t)][0],self.degree_hash_dict[(v,t)][1]

    def computation_entropy_tempurature(self,u,v,t,max_num = 50):
        if (u,v,t) in self.entropy_tempurature_dict.keys():
            return self.entropy_tempurature_dict[(u,v,t)]
        node_num, edge_num = self.current_graph_count(t)
        dv,neibor_v_dic = self.hash_degree(v,t)
        du,neibor_u_dic = self.hash_degree(u,t)   
        dv = max(dv,1)
        du = max(du,1)
        tempurature_difference = 0
        delta_J = 0
        delta_K = 0
        vnE_orignal = 0
        vnE_changed = 0
        k = 1.380649 # Boltzmann constant
        neibor_v_l = list(neibor_v_dic.keys())
        neibor_u_l = list(neibor_u_dic.keys())
        v_neighborhood_length = len(neibor_v_l)
        u_neighborhood_length = len(neibor_u_l)
        neighborhood_length =  v_neighborhood_length + u_neighborhood_length
        if neighborhood_length > max_num:
            idx_v = []
            idx_u = []
            N = range(neighborhood_length)
            idx = random.sample(N,max_num)
            neighbor_list = neibor_v_l + neibor_u_l
            for i in idx:
                if i < v_neighborhood_length:
                    idx_v.append(i)
                else:
                    idx_u.append(i)
            neibor_v_l = [ i for i in neighbor_list if i in idx_v]     
            neibor_u_l = [j for j in neighbor_list if j in idx_u] 
        def get_neighbor_dict(neighbor_dict,x,flag = 'False'):
            change = 0
            if flag == 'True':
                change = 1
            if x in neighbor_dict:
                return neighbor_dict[x] + change
            else:
                return change
        def compute_K(Auv,Auw,Avw,du,dv,dw):
            if du == 0 or dv == 0 or dw == 0:
                return 0
            else:
                return Auv*Auw*Avw/(du*dv*dw)
        def compute_J(Auv,du,dv):
            if du == 0 or dv == 0:
                return 0
            else:
                return Auv/(du*dv)
        def compute_vnE(Auv,du,dv):
            if du == 0 or dv == 0:
                return 0
            else:
                return Auv/(du*dv)
        def compute_delta_J(Aix,di,dx):
            if di == 0 or dx == 0:
                return 0
            else:
                return Aix/(di*dx*(dx+1))
        def compute_delta_K_part1(Aiu,Aiv,Auv,di,du,dv):
            if di == 0 or du == 0 or dv == 0:
                return 0
            else:
                return Aiu*Aiv*Auv/(di*du*dv*(dv+1)*(du+1))              
        def compute_delta_K_part2(Avi,Avj,Aiu,Aju,Aij,di,dj,du,dv):
            if du == 0 or dv == 0 or di == 0 or dj == 0:
                return 0
            else:
                return Avi*Avj*Aij/(di*dj*dv*(dv+1)) + Aiu*Aju*Aij/(di*dj*du*(du+1))

        for i,neighbor_i in enumerate(neibor_v_l+neibor_u_l):
            if neighbor_i in [u,v]:
                continue
            d_neighbor_i,neighbor_dict_i = self.hash_degree(neighbor_i,t)
            delta_J += compute_delta_J(Aix = get_neighbor_dict(neighbor_dict_i,v) , di = d_neighbor_i , dx = dv) 
            delta_J += compute_delta_J(Aix = get_neighbor_dict(neighbor_dict_i,u) , di = d_neighbor_i , dx = du)
            delta_K += compute_delta_K_part1(Aiu = get_neighbor_dict(neighbor_dict_i,u) , Aiv = get_neighbor_dict(neighbor_dict_i,v) , 
                                             Auv = get_neighbor_dict(neibor_v_dic,v) , di = d_neighbor_i , du = du , dv = dv)
                                             
            for j,neighbor_j in enumerate(neibor_v_l+neibor_u_l):
                if neighbor_j in [u,v]:
                    continue
                d_neighbor_j,neighbor_dict_j = self.hash_degree(neighbor_j,t)
                delta_K += compute_delta_K_part2(Avi = get_neighbor_dict(neighbor_dict_i,v) , Avj = get_neighbor_dict(neighbor_dict_j,v) , 
                                                Aiu = get_neighbor_dict(neighbor_dict_i,u) , Aju = get_neighbor_dict(neighbor_dict_j,u) , 
                                                Aij = get_neighbor_dict(neighbor_dict_i,neighbor_j) , 
                                                di = d_neighbor_i , dj = d_neighbor_j , du = du , dv = dv)
                vnE_orignal += compute_vnE( Auv = get_neighbor_dict(neighbor_dict_i,j) , du = du , dv = dv)

        
        dvc,duc = dv+1,du+1
        delta_K_final = delta_K 
        delta_J_final = delta_J 
        delta_J_final = delta_J_final + compute_J(   Auv = get_neighbor_dict(neibor_v_dic,v) , du = dvc , dv = dvc) \
                        -compute_J(   Auv = get_neighbor_dict(neibor_v_dic,v) , du = dv , dv = dv) 
        cache =     (compute_J(   Auv = get_neighbor_dict(neibor_v_dic,u,'True') , du = duc , dv = dvc) \
                        -compute_J(   Auv = get_neighbor_dict(neibor_v_dic,u) , du = du , dv = dv) )*2 
        delta_J_final = delta_J_final + cache
        delta_J_final = delta_J_final + compute_J(   Auv = get_neighbor_dict(neibor_u_dic,u) , du = duc , dv = duc) \
                        -compute_J(   Auv = get_neighbor_dict(neibor_u_dic,u) , du = du , dv = du)     

        vnE_orignal  += compute_vnE( Auv = get_neighbor_dict(neibor_v_dic,v) , du = dv , dv = dv) 
        vnE_orignal  += compute_vnE( Auv = get_neighbor_dict(neibor_u_dic,u) , du = du , dv = du) 

        vnE_changed = copy.deepcopy(vnE_orignal)
        vnE_orignal  += compute_vnE( Auv = get_neighbor_dict(neibor_v_dic,u) , du = du , dv = dv) 
        vnE_changed += compute_vnE( Auv = get_neighbor_dict(neibor_v_dic,u,'True') , du = duc , dv = dvc) 

        flagki,flagkj,flagij = 'True','True','True'         
        for i in [u,v]:
            for j in [u,v]:
                for k in [u,v]:
                    d_j,neighbor_dict_j = self.hash_degree(i,t) 
                    d_i,neighbor_dict_i = self.hash_degree(j,t) 
                    d_k,neighbor_dict_k = self.hash_degree(k,t)        
                    if k == i : 
                        flagki = 'False'
                    if i == j : 
                        flagij = 'False'
                    if k == j : 
                        flagkj = 'False'
                    delta_K_final += compute_K(get_neighbor_dict(neighbor_dict_k,i,flagki) , get_neighbor_dict(neighbor_dict_k,j,flagkj) ,get_neighbor_dict(neighbor_dict_i,j,flagij),
                                                d_k,d_i,d_j) 
                    delta_K_final -= compute_K(get_neighbor_dict(neighbor_dict_k,i) , get_neighbor_dict(neighbor_dict_k,j) ,get_neighbor_dict(neighbor_dict_i,j),
                                                d_k,d_i,d_j) 
                
        tempurature_difference = -2/k + 2/(3*k) * (delta_K_final / max(delta_J_final,10e-3) )
        vnE_orignal = 1 - 1 / node_num - (1 / (node_num**2) ) * vnE_orignal
        vnE_changed = 1 - 1 / node_num - (1 / (node_num**2) ) * vnE_changed
        expert_feature = np.zeros(([3])).astype(np.float32)
        expert_feature[0],expert_feature[1],expert_feature[2] = vnE_orignal, vnE_changed, tempurature_difference
        self.entropy_tempurature_dict[(v,u,t)] = expert_feature
        self.entropy_tempurature_dict[(u,v,t)] = expert_feature
        return  expert_feature
