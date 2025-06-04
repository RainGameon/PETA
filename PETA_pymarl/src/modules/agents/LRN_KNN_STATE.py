import numpy as np
from sklearn.neighbors import BallTree,KDTree
import os
import gc
import torch as th
import pickle
from sys import platform
import numpy as np

def inverse_distance(h, h_i, epsilon=1e-3):
    #return 1 / (th.dist(h, h_i) + epsilon)
    return 1 / ( np.linalg.norm( h - h_i ) + epsilon) # L2 Euclidean distance

class LRU_KNN_STATE:
    def __init__(self, capacity, state_dim, args, env_name, random_projection, state_embed_net=None):

        z_dim = args.emdqn_latent_dim

        self.env_name = env_name
        self.capacity = capacity
        self.n_agent= args.n_agents
        self.device = args.device
        self.flag_stats_norm = args.flag_stats_norm
        self.random_projection = random_projection
        self.state_embed_net = state_embed_net
        self.fixed_delta     = args.fixed_delta
        self.delta_cover_type = int(args.delta_cover_type)

        self.memory_emb_type  = int(args.memory_emb_type) # 1: random projection, 2: state itself

        self.atol  = args.atol_memory *  np.ones(1, dtype=np.float32)
        self.rtol  = args.rtol_memory *  np.ones(1, dtype=np.float32)
        self.atol_monitor = self.atol *  np.ones(1, dtype=np.float32)
        self.rtol_monitor = self.rtol *  np.ones(1, dtype=np.float32)
        self.mu_Ncall     = np.zeros(1, dtype=np.float32)
        self.mu_Nxi       = np.zeros(1, dtype=np.float32)
        self.mu_ratio_xi  = np.zeros(1, dtype=np.float32)
        self.z_dim = z_dim

        self.use_AEM = args.use_AEM

        # node information
        self.states         = np.empty((capacity, z_dim), dtype = np.float32) # projected value (z)
        self.states_norm    = np.empty((capacity, z_dim), dtype = np.float32) # y = (x- mu)/sigma
        self.global_states  = np.empty((capacity, state_dim), dtype = np.float32) # global state

        self.z_mu           = np.zeros(self.z_dim, dtype = np.float32)
        self.z_sigma        = np.ones(self.z_dim,  dtype = np.float32)
        self.x_mu           = np.zeros(self.z_dim, dtype = np.float32)
        self.x_sigma        = np.ones(self.z_dim,  dtype = np.float32)
        self.x_mu_monitor    = self.x_mu
        self.x_sigma_monitor = self.x_sigma 

        self.q_values_decay = np.zeros(capacity, dtype = np.float32) # = H(phi(s))
        self.tg             = np.zeros(capacity, dtype = int) # time step        
        self.xi             = np.zeros(capacity, dtype = np.uint)       # 记录xi[ind】是否为desirable state
        self.gamma          = args.gamma

        # cnt
        self.Ncall          = np.zeros(capacity, dtype = int) # 记录访问ind的次数，the number of transition (call)
        self.Nxi            = np.zeros(capacity, dtype = int) # 记录ind作为desirablethe state embedding次数 number of optimal transition
        #self.rcnt           = np.zeros(capacity, dtype = np.float32) # = H(phi(s))
        self.epsilon        = 0.001

        # obsolete
        self.kernel         = inverse_distance

        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None
        self.addnum = 0
        self.buildnum = 256
        self.buildnum_max = 256
        self.bufpath = './buffer/%s'%self.env_name
        self.build_tree_times = 0
        self.build_tree = False

    def update_states_norm(self):   
        if self.build_tree == False:
            return

        self.x_mu_monitor    = np.mean(self.states[:self.curr_capacity],axis=0)
        self.x_sigma_monitor = np.std(self.states[:self.curr_capacity] ,axis=0)
                  
        if self.flag_stats_norm == True:
            
            self.x_mu    = self.x_mu_monitor
            self.x_sigma = self.x_sigma_monitor

            for i in range(0, self.z_dim ):
                self.states_norm[:self.curr_capacity,i] = (self.states[:self.curr_capacity,i] - self.x_mu[i])/self.x_sigma[i]

            #.. compute states of state_norm
            self.z_mu    = np.mean(self.states_norm[:self.curr_capacity],axis=0)
            self.z_sigma = np.std(self.states_norm[:self.curr_capacity] ,axis=0)
            max_z_sigma  = max(self.z_sigma)            

            #.. tolerance update
            if self.delta_cover_type == 1:
                self.atol_monitor = np.power(2.0 * max_z_sigma, self.z_dim ) / self.capacity
                self.rtol_monitor = np.zeros(1, dtype = np.float32)
            elif self.delta_cover_type == 2:
                self.atol_monitor = np.power(2.0*3.0 * max_z_sigma, self.z_dim ) / self.capacity
                self.rtol_monitor = np.zeros(1, dtype = np.float32)

            if self.fixed_delta == False:
                self.atol = self.atol_monitor
                self.rtol = self.rtol_monitor
        else:
            self.states_norm = self.states
    
        #.. modified version ----------------------------------------------------------------------------------------------------------
    def peek_modified_EC(self, key, value_decay, xit, modify, global_state, cur_time):
        # input: key: global state
        # input: Rt, xi, modify
        # value_decay：return to go

        # output: H(key_hat), xi(key_hat) 

        if modify == False:
            checkpoint = 1

        if self.curr_capacity==0 or self.build_tree == False:
            return None, None, None
        # 寻找最近邻
        dist, ind = self.tree.query([key], k=1) # pick nearest one # 1-1 projection?  # TO CHECK: how about building tree based on states_norm and query ind with vector_atol ???
        # xt_hat
        ind = ind[0][0]

        # 【yt】 normalization
        key_norm = ((key - self.x_mu) / self.x_sigma) # check element-wise operation

        # 如果yt和yt_hat的距离足够近
        if np.allclose(self.states_norm[ind], key_norm, rtol=self.rtol, atol=self.atol ):
            self.lru[ind] = self.tm # update its updated time            
            self.tm +=0.01
            if modify:
                # 访问ind的次数+1
                self.Ncall[ind] += 1
                # 如果是desirable transition则次数+1
                if xit == 1:
                    self.Nxi[ind] += 1 # optimal transition
                # 如果是第一次访问到desirable state
                if (self.xi[ind] == 0) and (xit == 1) and self.use_AEM : # memory shift
                    self.xi[ind]          = xit                 # ind是desirable state embedding
                    self.states[ind]      = key                 # 更新xt
                    self.states_norm[ind] = key_norm            # 更新yt
                    self.global_states[ind]  = global_state     # 更新st
                    self.q_values_decay[ind] = value_decay      # 更新Ht，value_decay：return to go
                    self.Ncall[ind]       = 1                   # 第一次访问ind
                    self.Nxi[ind]         = 1                   # ind进行desirable transition次数
                    self.tg[ind]          = cur_time
                # 如果ind已经作为desirable state访问过，则进行最大回报更新
                else: # update Qval (value_decay: current Return/return to go)
                    if value_decay > self.q_values_decay[ind]: 
                        self.q_values_decay[ind] = value_decay
                                    
            rcnt = float(self.Nxi[ind] / (self.Ncall[ind] + self.epsilon))

            return self.q_values_decay[ind], float(self.xi[ind]), rcnt
        
        return None, None, None

    def add_modified_EC(self, key, value_decay, xi, global_state, cur_time):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.states_norm[old_index] = (key - self.x_mu)/self.x_sigma # check element-wise operation
            self.q_values_decay[old_index] = value_decay
            self.global_states[old_index]  = global_state
            self.xi[old_index] = xi
            self.lru[old_index] = self.tm
            self.Nxi[old_index]  = 0
            self.Ncall[old_index] = 0
            self.tg[old_index] = cur_time
        else:
            self.states[self.curr_capacity] = key
            self.states_norm[self.curr_capacity] = (key - self.x_mu)/self.x_sigma # check element-wise operation
            self.global_states[self.curr_capacity]  = global_state
            self.q_values_decay[self.curr_capacity] = value_decay
            self.xi[self.curr_capacity] = xi
            self.lru[self.curr_capacity] = self.tm
            self.Nxi[self.curr_capacity]  = 0
            self.Ncall[self.curr_capacity] = 0
            self.tg[self.curr_capacity] = cur_time
            self.curr_capacity+=1
        self.tm += 0.01

    #.. original version ----------------------------------------------------------------------------------------------------------
    def peek_EC(self, key, value_decay, modify):
        if modify == False:
            x = 1

        if self.curr_capacity==0 or self.build_tree == False:
            return None

        dist, ind = self.tree.query([key], k=1) # pick nearest one
        ind = ind[0][0]
                        
        if np.allclose(self.states[ind], key, rtol=self.rtol, atol=self.atol ):
            self.lru[ind] = self.tm # update its updated time
            self.tm +=0.01
            if modify:
                if value_decay > self.q_values_decay[ind]: # update Qval (value_decay: current Return)
                    self.q_values_decay[ind] = value_decay
            return self.q_values_decay[ind]
        #print self.states[ind], key

        return None

    def add_EC(self, key, value_decay):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values_decay[old_index] = value_decay
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.q_values_decay[self.curr_capacity] = value_decay
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity+=1
        self.tm += 0.01
    #---------------------------------------------------------------------------------------------------------------------------------------
    def update_kdtree(self):
        if self.build_tree:
            del self.tree
        self.tree = KDTree(self.states[:self.curr_capacity])
        self.build_tree = True
        self.build_tree_times += 1
        if self.build_tree_times == 50:
            self.build_tree_times = 0
            gc.collect()

    def check_tolerance(self):
        return self.atol, self.rtol, self.x_mu, self.x_sigma, self.z_mu, self.z_sigma

    def check_Ncall(self):
        self.mu_Ncall = np.mean(self.Ncall[:self.curr_capacity])
        self.mu_Nxi   = np.mean(self.Nxi[:self.curr_capacity])

        ratio_xi = np.divide( self.Nxi[:self.curr_capacity], self.Ncall[:self.curr_capacity] ) # element-wise
        self.mu_ratio_xi = np.mean(ratio_xi)

        return self.mu_Ncall, self.mu_Nxi, self.mu_ratio_xi, self.curr_capacity

    def save_memory(self, savepath):
        if not os.path.exists('buffer'):
            os.makedirs('buffer')
        if not os.path.exists(self.bufpath):
            os.makedirs(self.bufpath)

        np.save(os.path.join(savepath, 'states'), self.states[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'q_values_decay'), self.q_values_decay[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'Ncall'), self.Ncall[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'Nxi'), self.Nxi[:self.curr_capacity] )
        #np.save(os.path.join(savepath, 'rnd_projection'), self.random_projection )
        np.save(os.path.join(savepath, 'states_norm'), self.states_norm[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'global_states'), self.global_states[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'tg'), self.tg[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'lru'), self.lru[:self.curr_capacity] )
        np.save(os.path.join(savepath, 'xi'), self.xi[:self.curr_capacity] )

        #if platform == "linux":
        #    pkl_path = savepath + '/' + 'edge.pickle'
        #else:
        #    pkl_path = savepath + '\\' + 'edge.pickle'

        #with open(pkl_path, 'wb') as fw:
        #    pickle.dump(self.Edge, fw)

    def save(self, action):
        if not os.path.exists('buffer'):
            os.makedirs('buffer')
        if not os.path.exists(self.bufpath):
            os.makedirs(self.bufpath)
        np.save(os.path.join(self.bufpath, 'states_%d'%action), self.states[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'states_norm_%d'%action), self.states_norm[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'q_values_decay_%d'%action), self.q_values_decay[:self.curr_capacity])
        np.save(os.path.join(self.bufpath, 'lru_%d'%action), self.lru[:self.curr_capacity])

    def knn_value(self, key, knn):
        knn = min(self.curr_capacity, knn)
        if self.curr_capacity==0 or self.build_tree == False:
            return 0.0, 0.0

        dist, ind = self.tree.query([key], k=knn)

        value = 0.0
        value_decay = 0.0
        for index in ind[0]:
            value_decay += self.q_values_decay[index]
            self.lru[index] = self.tm
            self.tm+=0.01

        q_decay = value_decay / knn

        return q_decay

    def load(self, action):
        try:
            assert(os.path.exists(self.bufpath))
            lru = np.load(os.path.join(self.bufpath, 'lru_%d.npy'%action))
            cap = lru.shape[0]
            self.curr_capacity = cap
            self.tm = np.max(lru) + 0.01
            self.buildnum = self.buildnum_max

            self.states[:cap] = np.load(os.path.join(self.bufpath, 'states_%d.npy'%action))
            self.states_norm[:cap] = np.load(os.path.join(self.bufpath, 'states_norm_%d.npy'%action))
            self.q_values_decay[:cap] = np.load(os.path.join(self.bufpath, 'q_values_decay_%d.npy'%action))
            self.lru[:cap] = lru
            #self.tree = KDTree(self.states[:self.curr_capacity])
            self.tree = KDTree(self.states_norm[:self.curr_capacity])
            print ("load %d-th buffer success, cap=%d" % (action, cap))
        except:
            print ("load %d-th buffer failed" % action)

    def update_states_norm_old(self):        
        self.x_mu_monitor    = np.mean(self.states[:self.curr_capacity],axis=0)
        self.x_sigma_monitor = np.std(self.states[:self.curr_capacity] ,axis=0)
                
        max_x_sigma = max(self.x_sigma_monitor)            
        #min_x_sigma = min(self.x_sigma) 

        self.atol_monitor = np.power(2.0 * max_x_sigma, self.z_dim ) / self.capacity
        self.rtol_monitor = np.zeros(1, dtype = np.float32)
        
        if self.flag_stats_norm == True:
            self.atol = self.atol_monitor
            self.rtol = self.rtol_monitor
            self.x_mu    = self.x_mu_monitor
            self.x_sigma = self.x_sigma_monitor
            for i in range(0, self.z_dim ):
                self.states_norm[:self.curr_capacity,i] = (self.states[:self.curr_capacity,i] - self.x_mu[i])/self.x_sigma[i]
        else:
            self.states_norm = self.states