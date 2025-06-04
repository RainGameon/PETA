import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
from ..multiagentenv import MultiAgentEnv
import gym
import torch as th


class GoogleFootballEnv(MultiAgentEnv):

    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=0,
        render=False,
        num_agents=4,
        obs_dim=34,
        time_limit=200,
        time_step=0,
        map_name='academy_counterattack_hard',
        stacked=False,
        representation="simple115",
        rewards='scoring,checkpoints',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0,
    ):
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = num_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed
        
        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(self.seed)
        self.obs_dim = obs_dim # int(np.array(self.env.observation_space.shape[1:]).prod())  # TODO
        self.unit_dim = self.obs_dim  # TODO
        # self.state_dim = state_dim
        obs_space_low = self.env.observation_space.low[0]
        obs_space_high = self.env.observation_space.high[0]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.obs = None
    
    # TODO
    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False
    # 原来的
    '''
    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = _actions
        self.time_step += 1
        obs, rewards, done, infos = self.env.step(actions.tolist())

        self.obs = obs

        if self.time_step >= self.episode_limit:
            done = True

        return sum(rewards), done, infos
    '''
    # 使用CDS设计的稀疏奖励进行step
    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        if th.is_tensor(actions):
            actions = actions.cpu().numpy()
        else:
            actions = actions
        _, original_rewards, done, infos = self.env.step(actions.tolist())
        # rewards = list(original_rewards)
        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])
        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        if sum(original_rewards) <= 0:
            # return obs, self.get_global_state(), -int(done), done, infos
            return -int(done), done, infos

        # return obs, self.get_global_state(), 100, done, infos
        return 100, done, infos
    # MAT的step函数
    '''
    def step(self, actions):
        actions_int = [int(a) for a in actions]
        o, r, d, i = self.env.step(actions_int)
        obs = []
        ava = []
        for obs_dict in o:
            obs_i, ava_i = self._encode_obs(obs_dict)
            obs.append(obs_i)
            ava.append(ava_i)
        state = obs.copy()

        rewards = [[self.reward_encoder.calc_reward(_r, _prev_obs, _obs)]
                   for _r, _prev_obs, _obs in zip(r, self.pre_obs, o)]

        self.pre_obs = o

        dones = np.ones((self.n_agents), dtype=bool) * d
        infos = [i for n in range(self.n_agents)]
        return obs, state, rewards, dones, infos, ava
    '''
    # def get_obs(self):
    #     """Returns all agent observations in a list."""
    #     return self.obs.reshape(self.n_agents, -1)
    def get_obs(self):
        """Returns all agent observations in a list."""
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        # print(f'SMY —————————————————— obs.shape = {obs[0].shape}')
        return obs
   
   # def get_obs_agent(self, agent_id):
    #     """Returns observation for agent_id."""
    #     return self.obs[agent_id].reshape(-1)
    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)
    
    # def get_obs_size(self):
    #     """Returns the size of the observation."""
    #     obs_size = np.array(self.env.observation_space.shape[1:])
    #     return int(obs_size.prod())
    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team']
                              [-self.n_agents:].reshape(-1))
            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

            simple_obs.append(full_obs['right_team'].reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            simple_obs.append(
                (full_obs['right_team'] - ego_position).reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

        simple_obs = np.concatenate(simple_obs)

        return simple_obs

    # TODO
    # def get_global_state(self):
    #     state = self.get_simple_obs(-1)
    #     # print(f'SMY —————————————————— state.shape = {state.shape}')
    #     return state
    def get_global_state(self):
        return self.obs.flatten()

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    # def get_state_size(self):
    #     """Returns the size of the global state."""
    #     return self.get_obs_size() * self.n_agents
    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.obs_dim * self.n_agents # self.obs_dim state_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    # def reset(self):
    #     """Returns initial observations and states."""
    #     self.time_step = 0
    #     self.obs = self.env.reset()

    #     return self.get_obs(), self.get_global_state()
    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.env.reset()
        self.obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])

        return self.obs, self.get_global_state()
    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        return  {}
    
    # TODO
    def get_ally_alive(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ally_alive = np.zeros(self.n_agents)
        team_locs = cur_obs["left_team"][-self.n_agents:]  # 我方球员位置

        for i, pos in enumerate(team_locs):
            if np.abs(pos[0]) < 1e5:  # GRF中异常时球员坐标会变成极大值
                ally_alive[i] = 1
        return ally_alive

