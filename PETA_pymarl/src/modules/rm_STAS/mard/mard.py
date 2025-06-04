import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from modules.rm_STAS.mard.modules import EncoderLayer
# from modules.rm_STAS.mard.mard import STAS, rm_temporal
# from utils.norm import Normalizer

class ShapelyAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, n_agents, sample_num, device, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.n_agents = n_agents
        self.sample_num = sample_num
        self.phi = MultiAgentAttention(emb_dim, n_heads, n_agents, dropout, device)

        self.agent_embedding = nn.Embedding(self.n_agents, emb_dim)
    
    def get_attn_mask(self, shape):
        # use mask to estimate particular type of coalition
        mask = torch.bernoulli(torch.full((shape, shape), 0.5))
        mask = mask - torch.diag(torch.diag(mask)) + torch.eye(shape)

        return mask.to(self.device)

    def forward(self, input):
        """
        :param input: A (batch, number of agents, sequence length, emb dimension) tensor of input sequences.
        :return: deltas, the encoding of time-adjacent states and actions along the agents. (batch, number of agents, sequence length, action dimension).
        """
        b, n_a, t, e = input.size()
        input = input.permute(0, 2, 1, 3).contiguous().reshape(b*t, n_a, -1)
        coalition = np.arange(self.n_agents)
        np.random.shuffle(coalition)
        agent_embedding = self.agent_embedding(torch.tensor(coalition).to(self.device))[None, :, :].expand(b*t, n_a, self.emb_dim)
        input = input + agent_embedding
        shapley_reward = []

        for i in range(self.sample_num):
            attn_mask = self.get_attn_mask(n_a).unsqueeze(0).repeat(b*t, 1, 1)
            marginal_reward, _ = self.phi(input, input, input, attn_mask)
            shapley_reward.append(marginal_reward)

        shapley_reward = sum(shapley_reward)/self.sample_num
        shapley_reward = shapley_reward.reshape(b, t, n_a, -1).permute(0, 2, 1, 3)

        return shapley_reward


class rm_temporal(nn.Module):
    def __init__(self, input_dim, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents,
                 device, dropout=0.0, emb_dropout=0.5, action_space='discrete'):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.seq_length = seq_length
        self.device = device
        self.n_agents = n_agents
        self.emb_dropout = emb_dropout

        self.state_emb = nn.Linear(input_dim, emb_dim)
        if not action_space == 'discrete':
            self.action_emb = nn.Linear(input_dim, emb_dim)
        else:
            self.action_emb = nn.Embedding(n_actions + 1, emb_dim)
            self.joint_action_emb = nn.Linear(n_agents, emb_dim)

        self.pos_embedding = nn.Embedding(seq_length, emb_dim)

        self.layers = nn.ModuleList(
            [EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout) for _ in range(self.n_layer)])
        # self.shapley_block = ShapelyAttention(emb_dim, n_heads, self.n_agents, self.sample_num, device, emb_dropout)
        self.attention = nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=self.n_heads, dropout=emb_dropout)

        self.linear = nn.Linear(emb_dim, 1)
        self.linear2 = nn.Linear(emb_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
    def get_time_mask(self, episode_length,mask):
        # print(f'SMY: before mask={mask.shape}')
        mask = mask.squeeze()
        # mask = (torch.arange(self.seq_length)[None, :].to(self.device) < episode_length[:, None]).float()
        mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1, -2)
        # print(f'SMY: after mask={mask.shape}')
        return mask

    def forward(self, states, actions,mask, episode_length):
        # states = states.to(device)
        b, t, e = states.size()
        
        # positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, :, :, :].expand(b, n_a,  self.seq_length,  self.emb_dim)
        positions = self.pos_embedding(torch.arange(episode_length, device=self.device))[None, :, :].expand(b, episode_length, self.emb_dim)
        x1 = self.state_emb(states)
        # print(f'SMY: x1.shape = {x1.shape}')
        x2 = self.joint_action_emb(actions.squeeze().float()).view(b, t, -1)
        # print(f'SMY: x2.shape = {x2.shape}, positions = {positions.shape}')
        x = x1 + x2 + positions
        # x = self.state_emb(states) + self.action_emb(actions.squeeze()) + positions

        time_mask = self.get_time_mask(episode_length, mask)
        # time_mask = mask.squeeze()
        # x = x.permute(1, 0, 2)  # Shape: (time_steps, batch_size, emb_dim)
        # print(f'SMY: time_mask.shape = {time_mask.shape}, x = {x.shape}')

        # attn_output, _ = self.attention(x, x, x, key_padding_mask=time_mask)
        # print(f'smy: attn_output_0 = {attn_output.shape}')
        # Process the output with Transformer layers
        for layer in self.layers:
            x, _ = layer(x, time_mask)
            # print(f'smy: attn_output = {x.shape}')

        reward = self.linear(x).squeeze()
        reward_weight = self.softmax(self.linear2(x).squeeze() * mask.squeeze())
        weighted_reward = reward_weight * reward
        # print(f'SMY: reward={reward.shape}')
        return weighted_reward


class STAS(nn.Module):
    def __init__(self, input_dim, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
                device, dropout=0.0, emb_dropout=0.5, action_space='discrete'):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.seq_length = seq_length
        self.sample_num = sample_num
        self.device = device
        self.n_agents = n_agents
        self.emb_dropout = emb_dropout

        self.state_emb = nn.Linear(input_dim, emb_dim)
        if not action_space == 'discrete':
            self.action_emb = nn.Linear(input_dim, emb_dim)
        else:
            self.action_emb = nn.Embedding(n_actions+1, emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)

        self.layers = nn.ModuleList([EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout) for _ in range(self.n_layer)])
        self.shapley_block = ShapelyAttention(emb_dim, n_heads, self.n_agents, self.sample_num, device, emb_dropout)
        self.linear = nn.Linear(emb_dim, 1)

    def get_time_mask(self, episode_length):
        mask = (torch.arange(self.seq_length)[None, :].to(self.device) < episode_length[:, None]).float()
        mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1,-2)
        return mask

    def forward(self, states, actions, episode_length):
        b, n_a, t, e = states.size()

        positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, None, :, :].expand(b, n_a, self.seq_length, self.emb_dim)
        x = self.state_emb(states) + self.action_emb(actions).squeeze() + positions

        time_mask = self.get_time_mask(episode_length).repeat(n_a, 1, 1)
        x = x.reshape(b*n_a, t, -1).squeeze()
        for layer in self.layers:
            x, _ = layer(x, time_mask)

        shapley_reward = self.linear(self.shapley_block(x.reshape(b, n_a, t, -1))).squeeze()
        
        return shapley_reward

class STAS_ML(nn.Module):
    def __init__(self, input_dim, n_actions, emb_dim, n_heads, n_layer, seq_length, n_agents, sample_num,
                device, dropout=0.0, emb_dropout=0.5, action_space='discrete'):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.seq_length = seq_length
        self.sample_num = sample_num
        self.device = device
        self.n_agents = n_agents
        self.emb_dropout = emb_dropout

        self.state_emb = nn.Linear(input_dim, emb_dim)
        if not action_space == 'discrete':
            self.action_emb = nn.Linear(input_dim, emb_dim)
        else:
            self.action_emb = nn.Embedding(n_actions+1, emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)

        self.layers = nn.ModuleList([nn.ModuleList([EncoderLayer(self.emb_dim, self.n_heads, self.emb_dim, emb_dropout),
                                ShapelyAttention(emb_dim, n_heads, self.n_agents, self.sample_num, device, emb_dropout)]) for _ in range(self.n_layer)])
        self.shapley_block = ShapelyAttention(emb_dim, n_heads, self.n_agents, self.sample_num, device, emb_dropout)
        self.linear = nn.Linear(emb_dim*self.n_layer, 1)

    def get_time_mask(self, episode_length):
        mask = (torch.arange(self.seq_length)[None, :].to(self.device) < episode_length[:, None]).float()
        mask = torch.triu(torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))).transpose(-1,-2)
        return mask

    def forward(self, states, actions, episode_length):
        b, n_a, t, e = states.size()

        positions = self.pos_embedding(torch.arange(self.seq_length, device=self.device))[None, None, :, :].expand(b, n_a, self.seq_length, self.emb_dim)
        x = self.state_emb(states) + self.action_emb(actions).squeeze() + positions

        time_mask = self.get_time_mask(episode_length).repeat(n_a, 1, 1)
        x = x.reshape(b*n_a, t, -1).squeeze()
        shapley_rewards = []
        for layer in self.layers:
            x, _ = layer[0](x, time_mask)
            x = x.reshape(b, n_a, t, -1)
            x = layer[1](x)
            x = x.reshape(b*n_a, t, -1).squeeze()
            shapley_rewards.append(x)

        shapley_reward = self.linear(torch.cat(shapley_rewards, dim=-1).reshape(b, n_a, t, -1)).squeeze()
        
        return shapley_reward
