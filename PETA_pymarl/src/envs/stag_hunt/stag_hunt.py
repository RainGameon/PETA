# From https://github.com/wendelinboehmer/dcg/blob/master/src/envs/stag_hunt.py
from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert

'''
A simple n*m grid-world game for N agents trying to capture M prey and M' hares. 
No two entities can occupy the same position. The world can be either toroidal or bounded.
GRAPH INTERFACE
With the parameter state_as_graph=True, the .get_state() function returns a list of entities.
Each entity is represented as a dictionary with the following fields:
- 'type': {'agent', 'stag', 'hare'} denotes the type of entity
- 'pos': ndarray(2) denotes the position (vertical in [0=highest, n-1=lowest], horizontal in [0=leftest, m-1=rightest])
- 'avail_action': ndarray(5) only for 'type'='agent' denotes available actions (right, down, left, up, stay)
INTERSECTION OF OBSERVATIONS:
This version of predator prey also includes the observation method get_obs_intersection(agent_ids, global_view=False),
which returns a numpy array either as of size batch_size*state_dim (if global_view=True) or of size 
len(agent_ids)*batch_size*observation_dim (if global_view=False), where batch_size can be omitted if it is 
not specified in the constructor. The array contains the intersection of the observation of all agents 
in the list of agent_ids, either as state or as agent observation (centered around the respective agent),
but with the difference that the agent and prey (ID+1) are given, instead of just their presence (1).
Standard agent observations can be recovered by masking array>0.   
MOVEMENTS
Both predators and prey can move to the 4 adjacent states or remain in the current one. Movement is executed 
sequentially: first the predators move in a random order, then the prey chooses a random available action 
(i.e. an action that would not lead to a collision with another entity). 
A prey is captured if it cannot move (i.e. if 4 agents block all 4 adjacent fields).
REWARDS 
A captured prey is removed and yields a collaborative reward of +50. 
Forcing the a prey to move (scaring it off), by moving into the same field yields no additional reward. 
Collisions between agents is not punished, and each movement costs additional -0.1 reward. 
An episode ends if all prey have been captured.  
OBSERVATIONS
Prey only react to blocked movements (after the predators have moved), but predator agents observe all positions 
in a square of obs_size=(2*agent_obs+1) centered around the agent's current position. The observations are reshaped 
into a 1d vector of size (2*obs_size), including all predators and prey the agent can observe.
State output is a list of length 2, giving location of all agents and all targets.
TODO: Fine tune the reward to allow fast learning. 
'''

int_type = np.int16
float_type = np.float32


class StagHunt(MultiAgentEnv):

    action_labels = {'right': 0, 'down': 1, 'left': 2, 'up': 3, 'stay': 4, 'catch': 5,
                     'look-right': 6, 'look-down': 7, 'look-left': 8, 'look-up': 9}
    action_look_to_act = 6

    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        # TODO ！！！！！
        self.unit_dim = kwargs.get("unit_dim", 0)  # 假设默认值为 0  
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        self.print_caught_prey = getattr(args, "print_caught_prey", False)
        self.print_frozen_agents = getattr(args, "print_frozen_agents", False)

        # Add-on for graph interface
        self.state_as_graph = args.state_as_graph
        if self.state_as_graph:
            self.absolute_distance = getattr(args, "absolute_distance", False)
            self.normalise_distance = getattr(args, "normalise_distance", False)
            self.add_walls = getattr(args, "add_walls", False)
            self.prey_relational = getattr(args, "prey_relational", True)

        # Add-on for goat-hunts (which like to climb mountains)
        self.mountain_slope = getattr(args, "mountain_slope", 0.0)
        self.capture_conditions = getattr(args, "capture_conditions", [0, 1])
        self.mountain_spawn = getattr(args, "mountain_spawn", False)
        self.mountain_agent_row = getattr(args, "mountain_agent_row", -1)

        # Downwards compatibility of batch_mode
        self.batch_mode = batch_size is not None
        self.batch_size = batch_size if self.batch_mode else 1

        # Define the environment grid
        self.truncate_episodes = getattr(args, "truncate_episodes", True)
        self.observe_ids = getattr(args, "observe_ids", False)
        self.intersection_global_view = getattr(args, "intersection_global_view", False)
        self.intersection_unknown = getattr(args, "intersection_unknown", False)
        self.directed_observations = getattr(args, "directed_observations", False)
        self.directed_cone_narrow = getattr(args, "directed_cone_narrow", True)
        self.directed_exta_actions = getattr(args, "directed_exta_actions", True)
        self.random_ghosts = getattr(args, "random_ghosts", False)
        self.random_ghosts_prob = getattr(args, "random_ghosts_prob", 0.5)
        self.random_ghosts_mul = getattr(args, "random_ghosts_mul", -1.0)
        self.random_ghosts_random_indicator = getattr(args, "random_ghosts_indicator", False)
        self.observe_state = getattr(args, "observe_state", False)
        self.observe_walls = getattr(args, "observe_walls", True)
        self.observe_one_hot = getattr(args, "observe_one_hot", False)
        self.n_feats = (5 if self.observe_one_hot else 3) + (1 if self.random_ghosts else 0)
        self.toroidal = args.toroidal
        shape = args.world_shape
        self.x_max, self.y_max = shape
        self.state_size = self.x_max * self.y_max * self.n_feats
        self.env_max = np.asarray(shape, dtype=int_type)
        self.grid_shape = np.asarray(shape, dtype=int_type)
        self.grid = np.zeros((self.batch_size, self.x_max, self.y_max, self.n_feats), dtype=float_type)
        # 0=agents, 1=stag, 2=hare, [3=wall, 4=unknown], [-1=ghost-indicator]

        if self.random_ghosts:
            self.ghost_indicator = False        # indicator whether whether prey is a ghost (True) or not (False)
            self.ghost_indicator_potential_positions = np.asarray([[0, 0], [0, self.x_max-1], [self.y_max-1, 0],
                                                                   [self.y_max-1, self.x_max-1]], dtype=int_type)
            self.ghost_indicator_pos = [0, 0]   # position of the indicator whether prey is a ghost (-1) or not (+1)

        # Define the agents and their action space
        self.capture_action = getattr(args, "capture_action", False)
        self.capture_action_conditions = getattr(args, "capture_action_conditions", (2, 1))
        self.actions = np.asarray([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0], [0, 0],
                                   [0, 0], [0, 0], [0, 0], [0, 0]], dtype=int_type)
        self.action_names = ["right", "down", "left", "up", "stay", "catch",
                             'look-right', 'look-down', 'look-left', 'look-up']
        self.agent_move_block = np.asarray(getattr(args, "agent_move_block", [0]), dtype=int_type)
        self.n_actions = 10 if self.directed_observations and self.directed_exta_actions \
            else (6 if self.capture_action else 5)
        self.n_agents = args.n_agents
        self.n_stags = args.n_stags
        self.p_stags_rest = args.p_stags_rest
        self.n_hare = args.n_hare
        self.p_hare_rest = args.p_hare_rest
        self.n_prey = self.n_stags + self.n_hare
        self.agent_obs = args.agent_obs
        self.agent_obs_dim = np.asarray(self.agent_obs, dtype=int_type)

        if self.observe_state:
            # The size of the global state as observation (with one additional position feature)
            self.obs_size = int(self.state_size + self.grid_shape[0] * self.grid_shape[1])
        elif self.directed_observations and self.directed_cone_narrow:
            # The size of the visible observation cones for this option
            self.obs_size = self.n_feats * (2 * args.agent_obs[0] - 1) * (2 * args.agent_obs[1] - 1)
        else:
            # The agent-centric observation size
            self.obs_size = self.n_feats * (2 * args.agent_obs[0] + 1) * (2 * args.agent_obs[1] + 1)

        # Define the episode and rewards
        self.episode_limit = args.episode_limit
        self.time_reward = getattr(args, "reward_time", -0.1)
        self.collision_reward = getattr(args, "reward_collision", 0.0)
        self.capture_hare_reward = getattr(args, "reward_hare", 1.0)
        self.capture_stag_reward = getattr(args, "reward_stag", 2.0)
        self.miscapture_punishment = float(getattr(args, "miscapture_punishment", -self.capture_stag_reward))
        self.capture_terminal = getattr(args, "capture_terminal", True)
        self.capture_freezes = getattr(args, "capture_freezes", True)
        self.remove_frozen = getattr(args, "remove_frozen", False)

        # Define the internal state
        self.agents = np.zeros((self.n_agents, self.batch_size, 2), dtype=int_type)
        self.agents_not_frozen = np.ones((self.n_agents, self.batch_size), dtype=int_type)
        self.agents_orientation = np.zeros((self.n_agents, self.batch_size), dtype=int_type)  # use action_labels 0..3
        self.prey = np.zeros((self.n_prey, self.batch_size, 2), dtype=int_type)
        self.prey_alive = np.zeros((self.n_prey, self.batch_size), dtype=int_type)
        self.prey_type = np.ones((self.n_prey, self.batch_size), dtype=int_type)    # fill with stag (1)
        self.prey_type[self.n_stags:, :] = 2    # set hares to 2
        self.steps = 0
        self.sum_rewards = 0
        self.reset()

        self.made_screen = False
        self.scaling = 5

    # ---------- INTERACTION METHODS -----------------------------------------------------------------------------------
    def reset(self):
        # Reset old episode
        self.prey_alive.fill(1)
        self.agents_not_frozen.fill(1)
        self.steps = 0
        self.sum_rewards = 0

        # Clear the grid
        self.grid.fill(0.0)

        # Place n_agents and n_preys on the grid
        self._place_actors(self.agents, 0, row=self.mountain_agent_row if self.mountain_agent_row>= 0 else None)
        # Place the stags/goats
        self._place_actors(self.prey[:self.n_stags, :, :], 1, row=0 if self.mountain_spawn else None)
        # Place the hares/sheep
        self._place_actors(self.prey[self.n_stags:, :, :], 2, row=self.env_max[1]-1 if self.mountain_spawn else None)

        # Agent orientations are initialized randomly
        self.agents_orientation = np.random.random_integers(low=0, high=3, size=(self.n_agents, self.batch_size))

        if self.random_ghosts and self.random_ghosts_random_indicator:
            self.ghost_indicator_pos = self.ghost_indicator_potential_positions[
                random.randint(0, len(self.ghost_indicator_potential_positions)-1)].tolist()

        # self.step(th.zeros(self.n_agents).fill_(self.action_labels['stay']))
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Execute a*bs actions in the environment. """
        if not self.batch_mode:
            actions = np.expand_dims(np.asarray(actions, dtype=int_type), axis=1)
        assert len(actions.shape) == 2 and actions.shape[0] == self.n_agents and actions.shape[1] == self.batch_size, \
            "improper number of agents and/or parallel environments!"
        actions = actions.astype(dtype=int_type)

        # Initialise returned values and grid
        reward = np.ones(self.batch_size, dtype=float_type) * self.time_reward
        terminated = [False for _ in range(self.batch_size)]

        # Move the agents sequentially in random order
        for b in range(self.batch_size):
            for a in np.random.permutation(self.n_agents):
                # Only move if not frozen
                if self.agents_not_frozen[a, b] > 0:
                    # Only moves "up" if the mountain permits it (as defined by mountain_slope)
                    if not (np.random.rand() < self.mountain_slope and actions[a, b] == 3):
                        self.agents[a, b, :], collide = self._move_actor(self.agents[a, b, :], actions[a, b], b,
                                                                         self.agent_move_block, 0)
                        if collide:
                            reward[b] = reward[b] + self.collision_reward
                    # Set the agent's orientation (if the observation depends on it)
                    if self.directed_observations:
                        if self.directed_exta_actions:
                            if actions[a, b] >= self.action_look_to_act:
                                self.agents_orientation[a, b] = actions[a, b] - self.action_look_to_act
                        else:
                            if actions[a, b] < 4:
                                self.agents_orientation[a, b] = actions[a, b]

        # Move the prey
        for b in range(self.batch_size):
            for p in np.random.permutation(self.n_prey):
                if self.prey_alive[p, b] > 0:
                    # Collect all allowed actions for the prey
                    possible = []
                    next_to_agent = False
                    # Run through all potential movement actions (without actually moving)
                    for u in range(4):
                        if not self._move_actor(self.prey[p, b, :], u, b, np.asarray([0, 1, 2], dtype=int_type))[1]:
                            possible.append(u)
                        if self._move_actor(self.prey[p, b, :], u, b, np.asarray([0], dtype=int_type))[1]:
                            next_to_agent = True
                    # Capturing prey works differently when the agents have a 'catch' action
                    if self.capture_action:
                        n_catching_agents = 0
                        # Prey does not move (rest) if they have capture_conditions or less available moves
                        # if len(possible) <= self.capture_conditions[self.prey_type[p, b] - 1]:
                        #     possible = []
                        # Capturing happens if capture_action_conditions many agents execute 'catch'
                        for a in range(self.n_agents):
                            if actions[a, b] == self.action_labels['catch'] \
                                    and self.agents_not_frozen[a, b] > 0:
                                # If any movement action in [0, 4) would end up on agent a, that agent can 'catch' p
                                for u in range(4):
                                    pos = self.prey[p, b] + self.actions[u]
                                    if pos[0] == self.agents[a, b, 0] and pos[1] == self.agents[a, b, 1]:
                                        n_catching_agents += 1
                                        break
                        # If the number of neighboring agents that execute 'catch' >= condition, prey is captured
                        captured = n_catching_agents >= self.capture_action_conditions[self.prey_type[p, b] - 1]
                        #          and len(possible) == 0
                        if n_catching_agents > 0 and not captured:
                            reward[b] += self.miscapture_punishment
                    else:
                        # Prey is caught when the number of possible moves is less or equal to their capture_condition
                        captured = len(possible) <= self.capture_conditions[self.prey_type[p, b] - 1]
                    captured = captured and next_to_agent if self.args.prevent_cannibalism else captured
                    # If the prey is captured, remove it from the grid and terminate episode if specified
                    if captured:
                        # kill prey
                        self.prey_alive[p, b] = 0
                        self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], self.prey_type[p, b]] = 0
                        # terminate if capture_terminal=True
                        terminated[b] = terminated[b] or self.capture_terminal
                        # determine reward for capture
                        rew = 0
                        rew += self.capture_stag_reward if self.prey_type[p, b] == 1 else 0
                        rew += self.capture_hare_reward if self.prey_type[p, b] == 2 else 0
                        if self.random_ghosts and self.ghost_indicator:
                            rew *= self.random_ghosts_mul
                        reward[b] += rew
                        # freeze all surrounding agents if capture_freezes=True
                        if self.capture_freezes:
                            # each agent a...
                            for a in range(self.n_agents):
                                # ... which is not frozen...
                                if self.agents_not_frozen[a, b] > 0 \
                                        and (not self.capture_action or actions[a, b] == self.action_labels['catch']):
                                    # ... checks all possible movement actions ...
                                    for u in range(self.n_actions - 1):
                                        x = self.agents[a, b, :] + self.actions[u]
                                        # ... to see if it would have moved onto prey p's position ...
                                        if x[0] == self.prey[p, b, 0] and x[1] == self.prey[p, b, 1]:
                                            # ... which freezes the agent!
                                            self.agents_not_frozen[a, b] = 0
                                            # remove frozen agents from the grid if specified
                                            if self.remove_frozen:
                                                self.grid[b, self.agents[a, b, 0], self.agents[a, b, 1], 0] = 0
                                            # debug message if requested
                                            if self.print_frozen_agents:
                                                print("Freeze agent %u at height %u and pos %u." %
                                                      (a, self.env_max[0] - 1 - self.agents[a, b, 0], self.agents[a, b, 1]),
                                                      "    Agents active:", self.agents_not_frozen[:, b])
                        # print debug messages
                        if self.print_caught_prey:
                            print("Captured %s at time %u, height %d and pos %u." %
                                  (("stag" if self.prey_type[p, b] == 1 else "hare"), self.steps,
                                   self.env_max[0] - 1 - self.prey[p, b, 0], self.prey[p, b, 1]),
                                  "   Agents: ", self.agents_not_frozen[:, b],
                                  "   reward %g" % reward[b]
                                  )
                    else:
                        # If not, check if the prey can rest and if so determine randomly whether it wants to
                        rest = (self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], 0] == 0) \
                               and (np.random.rand() < (self.p_stags_rest if self.prey_type[p, b] == 1
                                                        else self.p_hare_rest)) \
                               or len(possible) == 0
                        # If the prey decides not to rest, choose a movement action randomly
                        if not rest:
                            u = possible[np.random.randint(len(possible))]
                            # Only moves up/down if the mountain permits it (as defined by mountain_slope)
                            if not (np.random.rand() < self.mountain_slope
                                    and self.grid[b, self.prey[p, b, 0], self.prey[p, b, 1], 0] == 0
                                    and (self.prey_type[p, b] == 2 and u == 3 or self.prey_type[p, b] == 1 and u == 1)):
                                # Execute movement
                                self.prey[p, b, :], _ = self._move_actor(self.prey[p, b, :], u, b,
                                                                         np.asarray([0, 1, 2], dtype=int_type),
                                                                         self.prey_type[p, b])
            # Terminate batch if all prey are caught or all agents are frozen
            terminated[b] = terminated[b] or sum(self.prey_alive[:, b]) == 0 or sum(self.agents_not_frozen[:, b]) == 0

        if self.random_ghosts:
            self.ghost_indicator = not (random.random() < self.random_ghosts_prob)

        # Terminate if episode_limit is reached
        info = {}
        self.sum_rewards += reward[0]
        self.steps += 1
        if self.steps >= self.episode_limit:
            terminated = [True for _ in range(self.batch_size)]
            info["episode_limit"] = self.truncate_episodes
        else:
            info["episode_limit"] = False

        if terminated[0] and self.print_caught_prey:
            print("Episode terminated at time %u with return %g" % (self.steps, self.sum_rewards))

        if self.batch_mode:
            return reward, terminated, info
        else:
            return reward[0].item(), int(terminated[0]), info

    # ---------- OBSERVATION METHODS -----------------------------------------------------------------------------------
    def get_obs_agent(self, agent_id, batch=0):
        if self.observe_state:
            # Get the state as observation (in the right format)
            dim = list(self.grid.shape)
            state = np.reshape(self.get_state(), dim)[batch, :]
            # Reshape and add a blank feature (last dimension) for the agent's position
            dim = dim[1:]   # only one batch
            dim[-1] += 1    # one more feature
            obs = np.zeros(dim)
            obs[:, :, :-1] = state
            # Mark the position of agent_id in the new feature
            obs[self.agents[agent_id, batch, 0], self.agents[agent_id, batch, 1], -1] = 1.0
            obs = obs.flatten()
        else:
            obs, _ = self._observe([agent_id])
        # If the frozen agents are removed, their observation is blank
        if self.capture_freezes and self.remove_frozen and self.agents_not_frozen[agent_id, batch] == 0:
            obs *= 0
        return obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state_as_graph(self):
        state = []
        # Enqueue all agents
        for a in range(self.n_agents):
            state.append({'type': 'agent', 'pos': self.agents[a, 0], 'avail_actions': [], 'id': a})
            # check the 4 movement actions for availability
            for u in range(self.n_actions - 1):
                _, c = self._move_actor(self.agents[a, 0, :], u, 0, np.asarray([0], dtype=int_type))
                state[-1]['avail_actions'].append(0 if c else 1)
            # 'pass' action (4) is always available
            state[-1]['avail_actions'].append(1)
        # Enqueue all stags
        for s in range(self.n_stags):
            state.append({'type': 'stag', 'pos': self.prey[s, 0]})
        # Enqueue all hares
        for h in range(self.n_hare):
            state.append({'type': 'hare', 'pos': self.prey[self.n_stags + h, 0]})
        # Return list f entities
        return state

    def get_state(self):
        # Either return the state as a list of entities...
        if self.state_as_graph:
            return self.state_to_graph(self.get_state_as_graph())
        # ... or return the entire grid
        if self.batch_mode:
            return self.grid.copy().reshape(self.state_size)
        else:
            return self.grid[0, :, :, :].reshape(self.state_size)

    def get_obs_intersect_pair_size(self):
        return 2 * self.get_obs_size()

    def get_obs_intersect_all_size(self):
        return self.n_agents * self.get_obs_size()

    def get_obs_intersection(self, agent_ids):
        return self._observe(agent_ids)

    # ---------- GETTERS -----------------------------------------------------------------------------------------------
    def get_total_actions(self):
        return self.n_actions

    def get_avail_agent_actions(self, agent_id):
        """ Currently runs only with batch_size==1. """
        if self.agents_not_frozen[agent_id] == 0:
            # All agents that are frozen can only perform the "stay" action
            avail_actions = [0 for _ in range(self.n_actions)]
            avail_actions[self.action_labels['stay']] = 1
        elif self.toroidal:
            # In a toroidal environment, all actions are allowed
            avail_actions = [1 for _ in range(self.n_actions)]
        else:
            # In a bounded environment, you cannot move into walls
            new_pos = self.agents[agent_id, 0, :] + self.actions[:self.n_actions]
            allowed = np.logical_and(new_pos >= 0, new_pos < self.grid_shape).all(axis=1)
            assert np.any(allowed), "No available action in the environment: this should never happen!"
            avail_actions = [int(allowed[a]) for a in range(self.n_actions)]
        # If the agent is not frozen, the 'catch' action is only available next to a prey
        if self.capture_action and self.agents_not_frozen[agent_id] > 0:
            avail_actions[self.action_labels['catch']] = 0
            # Check with virtual move actions if there is a prey next to the agent
            possible_catches = range(4) if not self.directed_observations \
                else range(self.agents_orientation[agent_id, 0], self.agents_orientation[agent_id, 0] + 1)
            for u in possible_catches:
                if self._move_actor(self.agents[agent_id, 0, :], u, 0, np.asarray([1, 2], dtype=int_type))[1]:
                    avail_actions[self.action_labels['catch']] = 1
                    break
        return avail_actions

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        return self.state_size

    def get_stats(self):
        pass

    def get_env_info(self):
        info = MultiAgentEnv.get_env_info(self)
        if self.state_as_graph:
            raise Exception
        return info

    # --------- RENDER METHODS -----------------------------------------------------------------------------------------
    def close(self):
        if self.made_screen:
            pygame.quit()
        print("Closing Multi-Agent Navigation")

    def render_array(self):
        # Return an rgb array of the frame
        return None

    def render(self):
        # TODO!
        pass

    def seed(self):
        raise NotImplementedError

# ---------- PRIVATE METHODS ---------------------------------------------------------------------------------------
    def _place_actors(self, actors: np.ndarray, type_id: int, row=None, col=None):
        for b in range(self.batch_size):
            for a in range(actors.shape[0]):
                is_free = False
                while not is_free:
                    # Draw actors's position randomly
                    actors[a, b, 0] = np.random.randint(self.env_max[0]) if row is None else row
                    actors[a, b, 1] = np.random.randint(self.env_max[1]) if col is None else col
                    # Check if position is valid
                    is_free = np.sum(self.grid[b, actors[a, b, 0], actors[a, b, 1], :]) == 0
                self.grid[b, actors[a, b, 0], actors[a, b, 1], type_id] = 1

    def print_grid(self, batch=0, grid=None):
        if grid is None:
            grid = self.grid
        grid = grid[batch, :, :, :].squeeze().copy()
        for i in range(grid.shape[2]):
            grid[:, :, i] *= i + 1
        grid = np.sum(grid, axis=2)
        print(grid)

    def print_agents(self, batch=0):
        obs = np.zeros((self.grid_shape[0], self.grid_shape[1]))
        for a in range(self.n_agents):
            obs[self.agents[a, batch, 0], self.agents[a, batch, 1]] = a + 1
        for p in range(self.n_prey):
            if self.prey_alive[p]:
                obs[self.prey[p, batch, 0], self.prey[p, batch, 1]] = -p - 1
        print(obs)

    def _env_bounds(self, positions: np.ndarray):
        # positions is 2 dimensional
        if self.toroidal:
            positions = positions % self.env_max
        else:
            positions = np.minimum(positions, self.env_max - 1)
            positions = np.maximum(positions, 0)
        return positions

    def _move_actor(self, pos: np.ndarray, action: int, batch: int, collision_mask: np.ndarray, move_type=None):
        # compute hypothetical next position
        new_pos = self._env_bounds(pos + self.actions[action])
        # check for a collision with anything in the collision_mask
        found_at_new_pos = self.grid[batch, new_pos[0], new_pos[1], :]
        collision = np.sum(found_at_new_pos[collision_mask]) > 0
        if collision:
            # No change in position
            new_pos = pos
        elif move_type is not None:
            # change the agent's state and position on the grid
            self.grid[batch, pos[0], pos[1], move_type] = 0
            self.grid[batch, new_pos[0], new_pos[1], move_type] = 1
        return new_pos, collision

    def _is_visible(self, agents, target):
        """ agents are plural and target is singular. """
        target = target.reshape(1, 2).repeat(agents.shape[0], 0)
        # Determine the Manhattan distance of all agents to the target
        if self.toroidal:
            # Account for the environment wrapping around in a toroidal fashion
            lower = np.minimum(agents, target)
            higher = np.maximum(agents, target)
            d = np.abs(np.minimum(higher - lower, lower - higher + self.grid_shape))
        else:
            # Account for the environment being bounded
            d = np.abs(agents - target)
        # Return true if all targets are visible by all agents
        return np.all(d <= self.agent_obs)

    def _intersect_targets(self, grid, agent_ids, targets, batch=0, target_id=0, targets_alive=None, offset=0):
        """" Helper for get_obs_intersection(). """
        for a in range(targets.shape[0]):
            marker = a + 1 if self.observe_ids else 1
            if targets_alive is None or targets_alive[a, batch]:
                # If the target is visible to all agents
                if self._is_visible(self.agents[agent_ids, batch, :], targets[a, batch, :]):
                    # include the target in all observations (in relative positions)
                    for o in range(len(agent_ids)):
                        grid[batch, targets[a, batch, 0] + offset, targets[a, batch, 1] + offset, target_id] = marker

    def _observe(self, agent_ids):
        # Compute available actions
        if len(agent_ids) == 1:
            avail_all = self.get_avail_agent_actions(agent_ids[0])
        elif len(agent_ids) == 2:
            a_a1 = np.reshape(np.array(self.get_avail_agent_actions(agent_ids[0])), [-1, 1])
            a_a2 = np.reshape(np.array(self.get_avail_agent_actions(agent_ids[1])), [1, -1])
            avail_actions = a_a1.dot(a_a2)
            avail_all = avail_actions * 0 + 1
        else:
            avail_all = []
        # Create over-sized grid
        ashape = np.array(self.agent_obs)
        ushape = self.grid_shape + 2 * ashape
        grid = np.zeros((self.batch_size, ushape[0], ushape[1], self.n_feats), dtype=float_type)
        # Make walls
        if self.observe_walls:
            wall_dim = 3 if self.observe_one_hot else 0
            wall_id = 1 if self.observe_one_hot else -1
            grid[:, :ashape[0], :, wall_dim] = wall_id
            grid[:, (self.grid_shape[0]+ashape[0]):, :, wall_dim] = wall_id
            grid[:, :, :ashape[1], wall_dim] = wall_id
            grid[:, :, (self.grid_shape[1] + ashape[1]):, wall_dim] = wall_id
        # Mark the ghost-indicator, if specified
        if self.random_ghosts:
            pos = [ashape[i] + self.ghost_indicator_pos[i] for i in range(2)]
            grid[0, pos[0], pos[1], -1] = -1 if self.ghost_indicator else 1
        # Mark the grid with all intersected entities
        noinformation = False
        for b in range(self.batch_size):
            if all([self._is_visible(self.agents[agent_ids, b, :], self.agents[agent_ids[a], b, :])
                    for a in range(len(agent_ids))]):
                # Every agent sees other intersected agents
                self._intersect_targets(grid, agent_ids, targets=self.agents, batch=b, target_id=0,
                                        targets_alive=self.agents_not_frozen, offset=ashape)
                # Every agent sees intersected stags
                self._intersect_targets(grid, agent_ids, targets=self.prey[:self.n_stags, :, :], batch=b, target_id=1,
                                        targets_alive=self.prey_alive[:self.n_stags, :], offset=ashape)
                # Every agent sees intersected hares
                self._intersect_targets(grid, agent_ids, targets=self.prey[self.n_stags:, :, :], batch=b, target_id=2,
                                        targets_alive=self.prey_alive[self.n_stags:, :], offset=ashape)
            else:
                noinformation = True
        # Mask out all unknown if specified
        if self.intersection_unknown:
            for b in range(self.batch_size):
                for a in agent_ids:
                    self._mask_agent(grid, self.agents[a, b, :] + ashape, ashape)

        if self.intersection_global_view:
            # In case of the global view
            obs = grid[:, ashape[0]:(ashape[0] + self.grid_shape[0]), ashape[1]:(ashape[1] + self.grid_shape[1]), :]
            obs = obs.reshape((1, self.batch_size, self.state_size))
        else:
            # otherwise local view
            obs = np.zeros((len(agent_ids), self.batch_size, 2*ashape[0]+1, 2*ashape[1]+1, self.n_feats),
                           dtype=float_type)
            for b in range(self.batch_size):
                for i, a in enumerate(agent_ids):
                    obs[i, b, :, :, :] = grid[b, (self.agents[a, b, 0]):(self.agents[a, b, 0] + 2*ashape[0] + 1),
                                              (self.agents[a, b, 1]):(self.agents[a, b, 1] + 2*ashape[1] + 1), :]
            obs = obs.reshape(len(agent_ids), self.batch_size, -1)

        # Final check: if not all agents can see each other, the mutual knowledge is empty
        if noinformation:
            if self.intersection_unknown:
                obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[2] // self.n_feats, self.n_feats)
                unknown_dim = 4 if self.observe_one_hot else 1
                unknown_id = 1 if self.observe_one_hot else -1
                obs.fill(0.0)
                obs[:, :, :, unknown_dim] = unknown_id
                obs = obs.reshape(obs.shape[0], obs.shape[1], self.n_feats * obs.shape[2])
            else:
                obs = 0 * obs

        # Mask out everything that is not in the cone, if directed_observations=True
        if self.directed_observations:
            obs = self._mask_invisible(obs, agent_ids)

        # Return considering batch-mode
        if self.batch_mode:
            return obs, avail_all
        else:
            return obs[:, 0, :].squeeze(), avail_all

    def _mask_agent(self, grid, pos, ashape):
        unknown_dim = 4 if self.observe_one_hot else 1
        unknown_id = 1 if self.observe_one_hot else -1
        grid[:, :(pos[0] - ashape[0]), :, :].fill(0.0)
        grid[:, :(pos[0] - ashape[0]), :, unknown_dim] = unknown_id
        grid[:, (pos[0] + ashape[0] + 1):, :, :].fill(0.0)
        grid[:, (pos[0] + ashape[0] + 1):, :, unknown_dim] = unknown_id
        grid[:, :, :(pos[1] - ashape[1]), :].fill(0)
        grid[:, :, :(pos[1] - ashape[1]), unknown_dim] = unknown_id
        grid[:, :, (pos[1] + ashape[1] + 1):, :].fill(0.0)
        grid[:, :, (pos[1] + ashape[1] + 1):, unknown_dim] = unknown_id

    def _mask_invisible(self, obs, agent_ids):
        """ Generates new observations from obs that only contain the visible cone. """
        narrow = 1 if self.directed_cone_narrow else 0
        dim = list(obs.shape[:2]) + [2 * i + 1 for i in self.agent_obs] + [self.n_feats]
        obs = np.reshape(obs, tuple(dim))
        vis = -np.ones((dim[0], dim[1], 2 * self.agent_obs[0] + 1 - 2 * narrow, 2 * self.agent_obs[1] + 1 - 2 * narrow, self.n_feats))
        for b in range(dim[1]):
            for i, a in enumerate(agent_ids):
                if self.agents_orientation[a, b] == self.action_labels['up']:
                    for j in range(self.agent_obs[0] + 1 - narrow):
                        vis[i, b, j, j:(vis.shape[3] - j), :] \
                            = obs[i, b, j, (j + narrow):(obs.shape[3] - j - narrow), :]
                elif self.agents_orientation[a, b] == self.action_labels['down']:
                    for j in range(self.agent_obs[0] + 1 - narrow):
                        vis[i, b, -j - 1, j:(vis.shape[3] - j), :] \
                            = obs[i, b, -j - 1, (j + narrow):(obs.shape[3] - j - narrow), :]
                elif self.agents_orientation[a, b] == self.action_labels['left']:
                    for j in range(self.agent_obs[0] + 1 - narrow):
                        vis[i, b, j:(vis.shape[2] - j), j, :] \
                            = obs[i, b, (j + narrow):(obs.shape[2] - j - narrow), j, :]
                elif self.agents_orientation[a, b] == self.action_labels['right']:
                    for j in range(self.agent_obs[0] + 1 - narrow):
                        vis[i, b, j:(vis.shape[2] - j), -j - 1, :] \
                            = obs[i, b, (j + narrow):(obs.shape[2] - j - narrow), -j - 1, :]
                else:
                    assert True, "Agent directions need to be 0..3!"
        return vis.reshape(dim[:2] + [-1])

    def _get_obs_from_grid(self, grid, agent_id, batch=0):
        """ OBSOLETE! """
        if self.toroidal:
            return self._get_obs_from_grid_troidal(grid, agent_id, batch)
        else:
            return self._get_obs_from_grid_bounded(grid, agent_id, batch)

    def _get_obs_from_grid_bounded(self, grid, agent_id, batch=0):
        """ Return a bounded observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent. Values outside the bounds of the grid are set to 0.
            OBSOLETE! """
        # Create the empty observation grid
        agent_obs = np.zeros((2*self.agent_obs[0]+1, 2*self.agent_obs[1]+1, 3), dtype=float_type)
        # Determine the unbounded limits of the agent's observation
        ul = self.agents[agent_id, batch, :] - self.agent_obs
        lr = self.agents[agent_id, batch, :] + self.agent_obs
        # Bound the limits to the grid
        bul = np.maximum(ul, [0, 0])
        blr = np.minimum(lr, self.grid_shape - 1)
        # Compute the ranges in x/y direction in the agent's observation
        bias = bul - ul
        aoy = [bias[0], blr[0] - bul[0] + bias[0]]
        aox = [bias[1], blr[1] - bul[1] + bias[1]]
        # Copy the bounded observations
        agent_obs[aoy[0]:(aoy[1]+1), aox[0]:(aox[1]+1), :] = grid[batch, bul[0]:(blr[0]+1), bul[1]:(blr[1]+1), :]
        return np.reshape(agent_obs, self.obs_size)

    def _get_obs_from_grid_troidal(self, grid, agent_id, batch=0):
        """ Return a wrapped observation for other agents' locations and targets, the size specified by observation
            shape, centered on the agent.
            OBSOLETE! """
        a_x, a_y = self.agents[agent_id, batch, :]
        o_x, o_y = self.agent_obs
        x_range = range((a_x - o_x), (a_x + o_x + 1))
        y_range = range((a_y - o_y), (a_y + o_y + 1))
        ex_grid = grid[batch, :, :, :].astype(dtype=float_type)
        agent_obs = ex_grid.take(x_range, 0, mode='wrap').take(y_range, 1, mode='wrap')
        return np.reshape(agent_obs, self.obs_size)

    def _get_obs_intersection_old(self, agent_ids):
        """ Returns the intersection of the all of agent_ids agents' observations.
            OBSOLETE, only maintained for legacy issues! """
        # Create grid
        grid = np.zeros((self.batch_size, self.grid_shape[0], self.grid_shape[0], 3), dtype=float_type)

        a_a1 = np.reshape( np.array(self.get_avail_agent_actions(agent_ids[0])),[-1,1])
        a_a2 = np.reshape( np.array(self.get_avail_agent_actions(agent_ids[1])),[1,-1])
        avail_actions = a_a1.dot(a_a2)
        avail_all = avail_actions * 0 + 1
        # If all agent_ids can see each other (otherwise the observation is empty)
        for b in range(self.batch_size):
            if all([self._is_visible(self.agents[agent_ids, b, :], self.agents[agent_ids[a], b, :])
                    for a in range(len(agent_ids))]):
                # Every agent sees other intersected agents
                self._intersect_targets(grid, agent_ids, targets=self.agents, batch=b, target_id=0)
                # Every agent sees intersected prey
                self._intersect_targets(grid, agent_ids, targets=self.prey, batch=b, target_id=1,
                                        targets_alive=self.prey_alive)
                avail_all = avail_actions
        # Return 0-1 encoded intersection if necessary
        if not self.observe_ids:
            grid = (grid != 0.0).astype(np.float32)
        # The intersection grid is constructed, now we have to generate the observations from it
        if self.intersection_global_view:
            # Return the intersection as a state
            if self.batch_mode:
                return grid.reshape((self.batch_size, self.state_size)), avail_all
            else:
                return grid[0, :, :, :].reshape(self.state_size), avail_all
        else:
            # Return the intersection as individual observations
            obs = np.zeros((len(agent_ids), self.batch_size, self.obs_size),
                           dtype=float_type)
            for b in range(self.batch_size):
                for a in range(len(agent_ids)):
                    obs[a, b, :] = self._get_obs_from_grid(grid, a, b)
            if self.batch_mode:
                return obs, avail_all
            else:
                return obs[:, 0, :], avail_all

    def state_to_graph(self, state_list):
        raise Exception
    #     return build_stag_hunt_env_graph(state_list, self.args)


    @classmethod
    def get_action_id(cls, label):
        return cls.action_labels[label]