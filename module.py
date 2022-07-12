import torch
import torch.nn as nn
import torch.nn.functional as F
from defend import Defense

class Net(nn.Module):
    def __init__(self, d1, d2, c, hidden, defense):
        super(Net, self).__init__()
        # d1 for passive party, d2 for active party
        self.d1 = d1
        self.d2 = d2
        self.input1 = nn.Linear(d1 - 1, hidden[0], bias=False)
        self.input1_sub = nn.Linear(d1, d1 - 1, bias=False)
        self.input2 = nn.Linear(d2, hidden[0], bias=True)
        hidden_layers = []
        for i in range(len(hidden) - 1):
            hidden_layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)
        self.final = nn.Linear(hidden[-1], c)
        self.inter = nn.Identity()
        self.defense = defense

    def forward(self, x):
        if True or isinstance(self.defense, Defense):
            x1 = self.input1_sub(x[:, :self.d1])
            #x1 = self.input1(torch.cat((x1, x[:, -1].reshape(-1, 1)), axis=1))
            x1 += self.defense(x1.detach(), x.detach(), self.input1.weight.detach())
            x1 = self.input1(x1)
        else:
            x1 = self.input1(x[:, :self.d1])
            x1 += self.defense(x1.detach(), x.detach(), self.input1.weight.detach())
            #print('gauss')
        x1 = self.inter(x1)
        x2 = self.input2(x[:, self.d1: self.d1 + self.d2])
        x = x1 + x2
        x = F.relu(x)
        x = self.hidden(x)
        x = self.final(x)
        return x

    def easy_forward(self, x):
        x1 = self.input1(x[:, :self.d1])
        x2 = self.input2(x[:, self.d1: self.d1 + self.d2])
        x = x1 + x2
        x = F.relu(x)
        x = self.hidden(x)
        x = self.final(x)
        return x
