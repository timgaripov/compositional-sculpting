# This code was adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py

from __future__ import print_function

import torch.nn.functional as F

from .util import *


class MNISTEncoder(nn.Module):
    def __init__(self, embed_dim=64, t_embed_dim=128, input_channels=1):
        super(MNISTEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216 + t_embed_dim, 512)
        self.fc2 = nn.Linear(512, embed_dim)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=t_embed_dim), nn.Linear(t_embed_dim, t_embed_dim))

    @staticmethod
    def act(x):
        return x * torch.sigmoid(x)

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(torch.cat([x, embed], dim=1))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class Classifier2ord(torch.nn.Module):
    def __init__(self, embed_dim=256, t_embed_tim=128, input_channels=1):
        super().__init__()
        self.trunk = MNISTEncoder(embed_dim=embed_dim, t_embed_dim=t_embed_tim, input_channels=input_channels)
        self.non_term_head = torch.nn.Linear(embed_dim, 2)

    def forward(self, x, t):
        # x: [batch_size, ndim * horizon]
        # terminal: [batch_size] 0.0 or 1.0
        x = self.trunk(x, t)
        non_term_outputs = self.non_term_head(x)

        # log_probs shape [batch_size, 2x2]
        # non-term probs:
        # p(y_1=1, y_2=1) = a
        # p(y_1=2, y_2=2) = b
        # p(y_1=1, y_2=2) = p(y_1=2, y_2=1) = c
        # a + b + 2c = 1
        # log(a + b + 2c) = 0
        # a = exp(o_0) / (exp(o_0) + exp(o_1) + 2 * 1)
        # b = exp(o_1) / (exp(o_0) + exp(o_1) + 2 * 1)
        # c = 1 / (exp(o_0) + exp(o_1) + 2 * 1)
        non_term_tmp = torch.cat([non_term_outputs, torch.full_like(non_term_outputs[:, :1], np.log(2.0))], dim=1)
        non_term_tmp = torch.log_softmax(non_term_tmp, dim=1)
        non_term_log_probs = torch.cat(
            [non_term_tmp[:, :1], non_term_tmp[:, 2:] - np.log(2.0), non_term_tmp[:, 2:] - np.log(2.0), non_term_tmp[:, 1:2]], dim=1
        )

        return non_term_log_probs.view(-1, 2, 2)
