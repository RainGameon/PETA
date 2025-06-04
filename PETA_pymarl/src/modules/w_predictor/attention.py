import torch
import torch.nn as nn
import torch.nn.functional as F
# ①基于每个agent的(oi,ai)送入rewardnet计算ri
# ②ri作为value，(oi,ai)作为query和key来进行attention


'''
input_size: obs_shape + n_actions
'''
class ModifiedSelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias=False)

        # self.rewards = nn.Linear(self.input_size, heads, bias=False)
        self.reward_predictor = nn.Linear(self.emb_size * heads, 512)
        self.reward_predictor2 = nn.Linear(512, 1)

    def forward(self, x):
        b, n, hin = x.size() # (b, t, state_shape + n_agent)
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'

        h = self.heads
        e = self.emb_size

        keys = self.tokeys(x).view(b, n, h, e)
        queries = self.toqueries(x).view(b, n, h, e)

        values = self.tovalues(x).view(b, n, h, e)
        # rewards = self.rewards(x).view(b, n, h, 1)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, n, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, n, e)
        # individual_rewards = torch.sum(rewards.clone(), dim = 2)  #(b,n,1)
        # rewards = rewards.transpose(1, 2).contiguous().view(b * h, n, 1)  # n个智能体的individual reward
        values = values.transpose(1, 2).contiguous().view(b * h, n, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, n, n)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)  # (b*h, n, n)
        self.dot = dot

        # out = torch.bmm(dot, rewards).view(b, h, n, 1)
        out = torch.bmm(dot, values).view(b, h, n, e)
        out = out.transpose(1, 2).contiguous().view(b, n, h*e)  # 每个agent考虑了与其他agent相关性后调整individual reward
        # 把不同head输出的individual reward看作是从不同角度获得的个人奖励，所以进行求和
        out2 = self.reward_predictor(out)
        rewards = self.reward_predictor2(out2)
        return rewards  #(b,t,1)
        # weighted_r = torch.sum(out, dim = 2)      #(b, n) 即（b，t）
        # total_r = torch.sum(weighted_ri, dim = 1)  #(b)
        # values = values.view(b, h, n, e)
        # values = values.transpose(1, 2).contiguous().view(b, n, h * e)
        # self.values = values
        # return weighted_r
        # total_r用于计算更新本网络的损失函数，rewards用于修正Qi送入QMIX
        # return total_r, individual_rewards
