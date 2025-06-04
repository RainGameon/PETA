import torch.nn as nn
import torch.nn.functional as F
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class rewardRNNet:
    def __init__(self, input_shape, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = input_shape
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args, predict_reward=True)

        self.hidden_states = None

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)  # [9, 1, 186]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, t, -1).transpose(1, 2)   # b, t, n, 1
    def init_hidden(self, batch_size):
        # hidden_states = torch.Size([1, 9, 64])
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        # print('SUNMENGYAO_________________mac: hidden_states_shape={}'.format(self.hidden_states.shape))
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
            bs = batch.batch_size
            inputs = []
            inputs.append(batch["obs"][:, :t])  # bTav
            inputs.append(batch["actions"][:, :t])  # bTa1?
            state_clone = batch["state"][:, :t].clone()
            state_input = state_clone.unsqueeze(-2).repeat(1, 1, self.args.n_agents, 1)  # （bs,T-1,n_agents,state_shape)
            inputs.append(state_input)

            inputs = th.cat([x.transpose(1, 2).reshape(bs * self.n_agents, t, -1) for x in inputs], dim=2)
            return inputs

    def parameters(self):
        return self.agent.parameters()

    def cuda(self):
        self.agent.cuda()

    def to(self, *args, **kwargs):
        self.agent.to(*args, **kwargs)
# ----------------------

# ①基于每个agent的(oi,ai)送入rewardnet计算ri
# ②ri作为value，(oi,ai)作为query和key来进行attention

'''
input_size: obs_shape + n_actions

class ModifiedSelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        self.tovalues = nn.Linear(1, self.emb_size * heads, bias=False)

        self.rewards = nn.Linear(self.input_size, heads, bias=False)

    def forward(self, x):
        b, n, hin = x.size() # (b*t, n_agent, obs_shape + n_actions)
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'

        h = self.heads
        e = self.emb_size

        keys = self.tokeys(x).view(b, n, h, e)
        queries = self.toqueries(x).view(b, n, h, e)

        # values = self.tovalues(x).view(b, n, h, e)
        rewards = self.rewards(x).view(b, n, h, 1)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, n, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, n, e)
        individual_rewards = torch.sum(rewards.clone(), dim = 2)  #(b,n,1)
        rewards = rewards.transpose(1, 2).contiguous().view(b * h, n, 1)  # n个智能体的individual reward
        # values = values.transpose(1, 2).contiguous().view(b * h, n, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, n, n)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)  # (b*h, n, n)
        self.dot = dot


        out = torch.bmm(dot, rewards).view(b, h, n, 1)
        out = out.transpose(1, 2).contiguous().view(b, n, h)  # 每个agent考虑了与其他agent相关性后调整individual reward
        # 把不同head输出的individual reward看作是从不同角度获得的个人奖励，所以进行求和
        weighted_ri = torch.sum(out, dim = 2)      #(b, n)
        total_r = torch.sum(weighted_ri, dim = 1)  #(b)
        # values = values.view(b, h, n, e)
        # values = values.transpose(1, 2).contiguous().view(b, n, h * e)
        # self.values = values

        # total_r用于计算更新本网络的损失函数，rewards用于修正Qi送入QMIX
        return total_r, individual_rewards
'''
