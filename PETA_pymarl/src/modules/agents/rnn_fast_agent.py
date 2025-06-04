import torch.nn as nn
import torch.nn.functional as F


class RNNFastAgent(nn.Module):
    def __init__(self, input_shape, args, predict_reward=False):
        super(RNNFastAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim) # input_shape=obs_shape + 1 + state_shape
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True
        )
        self.predict_reward = predict_reward
        if predict_reward:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        else:
            self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        # print('SUNMENGYAO_________________agent: rnn_hidden_dim={}'.format(self.args.rnn_hidden_dim))
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_() # rnn_hidden_dim=64

    def forward(self, inputs, hidden_state):
        bs = inputs.shape[0]  # 9
        epi_len = inputs.shape[1] # 1
        num_feat = inputs.shape[2] # 186
        # print('SUNMENGYAO_________________agent: inputs_shape = ', inputs.shape) # [9, 1, 186]
        inputs = inputs.reshape(bs * epi_len, num_feat) # [9,186],torch.float32
        # print('SUNMENGYAO_________________agent: inputs.device = ', inputs.device) # [9, 1, 186]
        # print('SUNMENGYAO_________________agent: fc1.weight.device = ', self.fc1.weight.device) # [9, 1, 186]
        # print('SUNMENGYAO_________________agent: inputs.dtype = ', inputs.dtype) # [9, 1, 186]
        # print('SUNMENGYAO_________________agent: fc1.weight.dtype = ', self.fc1.weight.dtype) # [9, 1, 186]

        x = self.fc1(inputs) # fc1.weight_shape =  torch.Size([64, 186], torch.float32
        x = F.relu(x)
        x = x.reshape(bs, epi_len, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(1, bs, self.args.rnn_hidden_dim).contiguous()
        x, h = self.rnn(x, h_in)
        x = x.reshape(bs * epi_len, self.args.rnn_hidden_dim)
        q = self.fc2(x)
        if self.predict_reward:
            q = q.reshape(bs, epi_len, 1)
        else:
            q = q.reshape(bs, epi_len, self.args.n_actions)
        return q, h
