import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
from utils.torch_utils import to_cuda
import numpy as np
from .vdn_Qlearner import vdn_QLearner
import os


class QPLEX_curiosity_vdn_Learner:
    def __init__(self, mac, scheme, logger, args, groups=None):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())

        #.. EMU ----------------------------------
        self.use_AEM         = args.use_AEM
        self.memory_emb_type = args.memory_emb_type            
        #-----------------------------------------

        self.last_target_update_episode = 0
        self.save_buffer_cnt = 0
        if self.args.save_buffer:
            self.args.save_buffer_path = os.path.join(self.args.save_buffer_path, str(self.args.seed))

        self.mixer = None
        self.vdn_learner = vdn_QLearner(mac, scheme, logger, args)
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.save_buffer_cnt = 0

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,intrinsic_rewards,
                  show_demo=False, save_data=None, show_v=False, save_buffer=False,ec_buffer=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] 
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        mac.init_hidden(batch.batch_size)
        mac_out = mac.forward(batch, batch.max_seq_length, batch_inf=True) # [batch, time, agent, actions]

        if save_buffer:
            curiosity_r=intrinsic_rewards.clone().detach().cpu().numpy()
            # rnd_r = rnd_intrinsic_rewards.clone().detach().cpu().numpy()
            # extrinsic_mac_out_save=extrinsic_mac_out.clone().detach().cpu().numpy()
            mac_out_save = mac_out.clone().detach().cpu().numpy()
            actions_save=actions.clone().detach().cpu().numpy()
            terminated_save=terminated.clone().detach().cpu().numpy()
            state_save=batch["state"][:, :-1].clone().detach().cpu().numpy()
            data_dic={'curiosity_r':curiosity_r,
                                 # 'extrinsic_Q':extrinsic_mac_out_save,
                        'control_Q':mac_out_save,'actions':actions_save,'terminated':terminated_save,
                        'state':state_save}

            self.save_buffer_cnt += self.args.save_buffer_cycle



            if not os.path.exists(self.args.save_buffer_path):
                os.makedirs(self.args.save_buffer_path)
            np.save(self.args.save_buffer_path +"/"+ 'data_{}'.format(self.save_buffer_cnt), data_dic)
            print('save buffer ({}) at time{}'.format(batch.batch_size, self.save_buffer_cnt))
            return

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, batch.max_seq_length, batch_inf=True)[:, 1:, ...]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = to_cuda(th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)), self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = \
                    mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv, _, _ = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                      max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
                ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                max_q_i=max_action_qvals, is_v=False)
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                         actions=cur_max_actions_onehot,
                                                         max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                    target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                   actions=cur_max_actions_onehot,
                                                   max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # Calculate 1-step Q-Learning targets
        if self.args.use_emdqn:
            ec_buffer.update_counter += 1
            qec_input = chosen_action_qvals.clone().detach()
            qec_input_new = []
            eta  = th.zeros_like(qec_input).detach().to(self.args.device)

            if self.use_AEM == False: # EMC
                for i in range(self.args.batch_size): # batch = 32
                    qec_tmp = qec_input[i, :]
                    for j in range(1, batch.max_seq_length):
                        if not mask[i, j - 1]:
                            continue           
                        ec_buffer.update_counter_call += 1 
                        z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())
                        q = ec_buffer.peek_EC(z, None, modify=False)
                        if q != None:
                            qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1]
                            ec_buffer.qecwatch.append(q)
                            ec_buffer.qec_found += 1
                    qec_input_new.append(qec_tmp)
                qec_input_new = th.stack(qec_input_new, dim=0)
                '''———————————————————————————————————EMU—————————————————————————————————————'''
            else: # EMU
                Vopt = target_max_qvals.clone().detach() # default value # start from s[t+1]               

                for i in range(self.args.batch_size): # batch = 32
                    qec_tmp = qec_input[i, :]
                    for j in range(1, batch.max_seq_length):
                        if not mask[i, j - 1]:
                            continue                 

                        ec_buffer.update_counter_call += 1 
                        if self.memory_emb_type == 1:
                            z = np.dot(ec_buffer.random_projection, batch["state"][i][j].cpu())
                        elif self.memory_emb_type == 2:
                            z = ec_buffer.state_embed_net(batch["state"][i][j].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy() # can be improved via batch-run
                        elif self.memory_emb_type == 3:
                            timestep = th.tensor( [float(j) / float(self.args.episode_limit)] ).to(self.args.device)
                            embed_input = th.cat( [ batch["state"][i][j], timestep], dim=0).unsqueeze(0).unsqueeze(0)

                            if self.args.encoder_type == 1: # FC
                                z = ec_buffer.state_embed_net( embed_input ).squeeze(0).squeeze(0).detach().cpu().numpy() # can be improved via batch-run
                            elif self.args.encoder_type == 2: # cVAE
                                mu, log_var = ec_buffer.state_embed_net( embed_input ) # can be improved via batch-run 
                                z = ec_buffer.reparameterize(mu, log_var, flagTraining=False).squeeze(0).squeeze(0).detach().cpu().numpy()

                        q, xi, rcnt = ec_buffer.peek_modified(z, None, 0, modify=False, global_state=None, cur_time=0)
                        
                        if q != None:
                            qec_tmp[j - 1] = self.args.gamma * q + rewards[i][j - 1]
                            ec_buffer.qecwatch.append(q)
                            ec_buffer.qec_found += 1
                            Vopt[i][j-1][0] = th.tensor(q).to(self.args.device)
                                                    
                        if self.args.optimality_type == 1 and rcnt !=None: # expected
                            eta[i][j-1] = rcnt * max(Vopt[i][j-1] - target_max_qvals[i][j-1], 0.0)
                        elif self.args.optimality_type == 2 and xi != None : # optimistic
                            eta[i][j-1] = xi * max(Vopt[i][j-1] - target_max_qvals[i][j-1], 0.0)
                        
                    qec_input_new.append(qec_tmp)
                qec_input_new = th.stack(qec_input_new, dim=0)
            
            episodic_q_hit_pro = 1.0 * ec_buffer.qec_found / self.args.batch_size / ec_buffer.update_counter / batch.max_seq_length
            episodic_qec_hit_pro_norm =  ec_buffer.qec_found /  ec_buffer.update_counter_call
                    
        #targets = float(self.args.optimality_incentive)*self.args.gamma*eta + intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        
        if self.args.optimality_incentive:
            targets = self.args.gamma*eta + intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        
        else:
            targets = intrinsic_rewards+rewards + self.args.gamma * (1 - terminated) * target_max_qvals        

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)
        if self.args.use_emdqn:
            emdqn_td_error = qec_input_new.detach() - chosen_action_qvals          
            #emdqn_masked_td_error = emdqn_td_error * mask * (1-float(self.args.optimality_incentive))
            if self.args.optimality_incentive:
                emdqn_masked_td_error = emdqn_td_error * mask * 0.0
            else:
                emdqn_masked_td_error = emdqn_td_error * mask

        if show_v:
            mask_elems = mask.sum().item()

            actual_v = rewards.clone().detach()
            for t in reversed(range(rewards.shape[1] - 1)):
                actual_v[:, t] += self.args.gamma * actual_v[:, t + 1]
            self.logger.log_stat("test_actual_return", (actual_v * mask).sum().item() / mask_elems, t_env)

            self.logger.log_stat("test_q_taken_mean", (chosen_action_qvals * mask).sum().item() / mask_elems, t_env)
            return

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
            if self.args.use_emdqn:
                emdqn_loss = (emdqn_masked_td_error ** 2).sum() / mask.sum() * self.args.emdqn_loss_weight
                loss += emdqn_loss

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval: # learner_log_interval 10000
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()

            if self.args.use_emdqn:                
                self.logger.log_stat("emdqn_loss", emdqn_loss.item(), t_env)
                self.logger.log_stat("emdqn_curr_capacity", ec_buffer.ec_buffer.curr_capacity, t_env)
                self.logger.log_stat("emdqn_weight", self.args.emdqn_loss_weight, t_env)

            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            
            self.log_stats_t = t_env

        if self.args.is_prioritized_buffer:
            return masked_td_error ** 2, mask
    # td_error = learner.train(episode_sample, runner.t_env, episode,ec_buffer=ec_buffer)
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None, show_v=False, ec_buffer=None):

        intrinsic_rewards = self.vdn_learner.train(batch, t_env, episode_num,save_buffer=False, imac=self.mac, timac=self.target_mac)
        if self.args.is_prioritized_buffer:
            masked_td_error, mask = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)
        else:
            self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards,
                           show_demo=show_demo, save_data=save_data, show_v=show_v,ec_buffer=ec_buffer)

        if hasattr(self.args, 'save_buffer') and self.args.save_buffer:
            if self.buffer.episodes_in_buffer - self.save_buffer_cnt >= self.args.save_buffer_cycle:
                if self.buffer.can_sample(self.args.save_buffer_cycle):
                    batch_tmp=self.buffer.sample(self.args.save_buffer_cycle, newest=True)
                    intrinsic_rewards_tmp=self.vdn_learner.train(batch_tmp, t_env, episode_num, save_buffer=True,
                                                                   imac=self.mac, timac=self.target_mac)
                    self.sub_train(batch_tmp, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,intrinsic_rewards=intrinsic_rewards_tmp,
                        show_demo=show_demo, save_data=save_data, show_v=show_v, save_buffer=True)


                else:
                    print('**' * 20, self.buffer.episodes_in_buffer, self.save_buffer_cnt)

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets(ec_buffer)
         
            self.last_target_update_episode = episode_num
            
        if self.args.is_prioritized_buffer:
            res = th.sum(masked_td_error, dim=(1, 2)) / th.sum(mask, dim=(1, 2))
            res = res.cpu().detach().numpy()
            return res

    def _update_targets(self,ec_buffer):
        if self.args.use_emdqn:
            ec_buffer.update_kdtree()
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        to_cuda(self.mac, self.args.device)
        to_cuda(self.target_mac, self.args.device)
        self.vdn_learner.cuda()
        if self.mixer is not None:
            to_cuda(self.mixer, self.args.device)
            to_cuda(self.target_mixer, self.args.device)

    def save_models(self, path, ec_buffer): ## save models from here...
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.args.use_AEM == True and self.args.use_emdqn == True and ( self.args.memory_emb_type == 2 or self.args.memory_emb_type == 3 ):
            th.save(ec_buffer.predict_mac.state_dict(), "{}/predict_mac.th".format(path))
            th.save(ec_buffer.state_embed_net.state_dict(), "{}/state_embed_net.th".format(path))

        #.. save model related to episodic memory
        if (ec_buffer is not None) and self.args.save_memory_info:
            if self.use_AEM: 
                ec_buffer.ec_buffer.save_memory(path)

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
