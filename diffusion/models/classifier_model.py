# This code was adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import *

class MNISTEncoder(nn.Module):
    def __init__(self, embed_dim = 64, t_embed_dim = 128, input_channels=1, channels=[32,64]):
        super(MNISTEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, 1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12 * 12 * channels[1] + t_embed_dim, 512)
        self.fc2 = nn.Linear(512, embed_dim)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=t_embed_dim),
         nn.Linear(t_embed_dim, t_embed_dim))
        self.act = lambda x: x * torch.sigmoid(x)

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

class JointYClassifier(torch.nn.Module):
    def __init__(self, embed_dim = 256, t_embed_tim = 128, input_channels=1):
        super().__init__()
        self.trunk = MNISTEncoder(embed_dim=embed_dim, t_embed_dim=t_embed_tim, input_channels=input_channels)
        self.non_term_head = torch.nn.Linear(embed_dim, 2)

    def forward(self, x, t):
        # x: [batch_size, ndim * horizon]
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
        non_term_log_probs = torch.cat([non_term_tmp[:, :1], non_term_tmp[:, 2:] - np.log(2.0),
                                        non_term_tmp[:, 2:] - np.log(2.0), non_term_tmp[:, 1:2]], dim=1)

        return non_term_log_probs.view(-1, 2, 2)
    

class ThreeWayJointYClassifier(torch.nn.Module):
    def __init__(self, embed_dim = 512, t_embed_tim = 128, input_channels=1):
        super().__init__()
        self.trunk = MNISTEncoder(embed_dim=embed_dim, t_embed_dim=t_embed_tim, input_channels=input_channels, channels=[64,96])
        self.head = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 5))

    def forward(self, x, t):
        # x: [batch_size, ndim * horizon]
        x = self.trunk(x, t)
        outputs = self.head(x)

        # log_probs shape [batch_size, 3x3]
        # p(y_1=1, y_2=1) = a
        # p(y_1=2, y_2=2) = b
        # p(y_1=3, y_2=3) = c
        # p(y_1=1, y_2=2) = d
        # p(y_1=1, y_2=3) = e
        # p(y_1=2, y_2=3) = f

        # a + b + c + 2*d + 2*e + 2*f = 1
        # a = exp(o_0) / (exp(o_0) + exp(o_1) + exp(o_2) + 2 * exp(o_3) + 2 * exp(o_4) + 2 * 1.0)
        # b = exp(o_1) / (exp(o_0) + exp(o_1) + exp(o_2) + 2 * exp(o_3) + 2 * exp(o_4) + 2 * 1.0)
        # c = exp(o_2) / (exp(o_0) + exp(o_1) + exp(o_2) + 2 * exp(o_3) + 2 * exp(o_4) + 2 * 1.0)
        # d = exp(0_3) / (exp(o_0) + exp(o_1) + exp(o_2) + 2 * exp(o_3) + 2 * exp(o_4) + 2 * 1.0)
        # e = exp(o_4) / (exp(o_0) + exp(o_1) + exp(o_2) + 2 * exp(o_3) + 2 * exp(o_4) + 2 * 1.0)
        # f = 1.0      / (exp(o_0) + exp(o_1) + exp(o_2) + 2 * exp(o_3) + 2 * exp(o_4) + 2 * 1.0)
        tmp = torch.cat([outputs, torch.full_like(outputs[:, :1], 0.0)], dim=1)
        tmp = tmp.add(torch.tensor([0.0, 0.0, 0.0, np.log(2.0), np.log(2.0), np.log(2.0)], dtype = x.dtype, device=x.device))
        tmp = torch.log_softmax(tmp, dim=1)
        log_probs = torch.cat([tmp[:, 0, None], tmp[:, 3, None] - np.log(2.0), tmp[:, 4, None] - np.log(2.0), tmp[:, 3, None] - np.log(2.0), tmp[:, 1, None], tmp[:, 5, None] - np.log(2.0), tmp[:, 4, None] - np.log(2.0), tmp[:, 5, None] - np.log(2.0), tmp[:, 2, None]], dim=1)
        return log_probs.view(-1, 3, 3)
    

class MNISTConditionalEncoder(nn.Module):
    def __init__(self, embed_dim = 64, t_embed_dim = 128, y_embed_dim = 6, input_channels=1, channels=[32,64]):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, 1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12 * 12 * channels[1] + t_embed_dim + y_embed_dim, 512)
        self.fc2 = nn.Linear(512 + y_embed_dim, embed_dim)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=t_embed_dim),
         nn.Linear(t_embed_dim, t_embed_dim))
        self.act = lambda x: x * torch.sigmoid(x)
            

    def forward(self, x, t, y_embed):
        embed = self.act(self.embed(t))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(torch.cat([x, embed, y_embed], dim=1))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(torch.cat([x, y_embed], dim=1))
        return x
    
class ThreeWayConditionalYClassifier(torch.nn.Module):
    def __init__(self, embed_dim = 256, t_embed_tim = 128, input_channels=1):
        super().__init__()
        self.trunk = MNISTConditionalEncoder(embed_dim=embed_dim, t_embed_dim=t_embed_tim, y_embed_dim=6, input_channels=input_channels, channels=[64,64])
        self.head = nn.Linear(256, 2)

    def embed_y(self, y_1_vec, y_2_vec, dtype, device):
        labels = []
        for (y_1, y_2) in zip(y_1_vec, y_2_vec):
            if (y_1 == 1) and (y_2 == 1):
                labels.append(0)
            elif (y_1 == 2) and (y_2 == 2):
                labels.append(1)
            elif (y_1 == 3) and (y_2 == 3):
                labels.append(2)
            elif ((y_1 == 1) and (y_2 == 2)) or ((y_1 == 2) and (y_2 == 1)):
                labels.append(3)
            elif ((y_1 == 1) and (y_2 == 3)) or ((y_1 == 3) and (y_2 == 1)):
                labels.append(4)
            elif ((y_1 == 3) and (y_2 == 2)) or ((y_1 == 2) and (y_2 == 3)):
                labels.append(5)
        return F.one_hot(torch.LongTensor(labels), num_classes=6).to(dtype=dtype, device=device)

    def forward(self, x, t, y_1, y_2):
        # x: [batch_size, ndim * horizon]
        y_embed = self.embed_y(y_1, y_2, x.dtype, x.device)
        x = self.trunk(x, t, y_embed)
        outputs = self.head(x)

        # log_probs shape [batch_size, 3]
        # p(y_3=1) = a
        # p(y_3=2) = b
        # p(y_3=3) = 1 - a - b

        # a + b + c + 2*d + 2*e + 2*f = 1
        # a = exp(o_0) / (exp(o_0) + exp(o_1) + 1.0)
        # b = exp(o_1) / (exp(o_0) + exp(o_1) + 1.0)
        # c = 1.0 / (exp(o_0) + exp(o_1) + 1.0)
        tmp = torch.cat([outputs, torch.full_like(outputs[:, :1], 0.0)], dim=1)
        return torch.log_softmax(tmp, dim=1)