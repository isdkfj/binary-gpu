import torch
import numpy as np
from utils import print_stat

class Gaussian:
    def __init__(self, eps, use_gpu=False):
        self.eps = eps
        self.generator = torch.Generator(device='cuda' if use_gpu else 'cpu')

    def defense(self, x1, *args):
        return torch.randn(x1.size(), generator=self.generator, device=x1.device) * self.eps

    def print_info(self, train_acc, test_acc, attack_acc):
        print('gaussian noise with eps = ', self.eps)
        print_stat('train_acc', train_acc)
        print_stat('test_acc', test_acc)
        print_stat('attack_acc', attack_acc)
        print('')

class Defense:
    def __init__(self, d1, binary_features):
        self.d1 = d1
        self.binary_features = binary_features

    def defense(self, x1, x, W):
        invW = torch.linalg.inv(W[:self.d1, :].T)
        Q = torch.linalg.solve(W[:self.d1, :].T, W.T)
        w = torch.mean(invW[:, self.binary_features], axis=1)
        # construct quadratic programming
        mat = torch.zeros((self.d1 + 1, self.d1 + 1), device=x1.device)
        mat[:self.d1, :self.d1] = 2 * Q @ Q.T
        mat[:self.d1, -1] = w
        mat[-1, :self.d1] = w
        vec = torch.zeros(self.d1 + 1, device=x1.device)
        vec[-1] = 1
        sol = torch.linalg.solve(mat, vec)
        #r = x[:, -1].reshape(-1, 1) - x1[:, :self.d1] @ w.reshape(-1, 1)
        r = torch.randint(2, (x.shape[0], 1)) - x1[:, :self.d1] @ w.reshape(-1, 1)
        r = r @ sol[:self.d1].reshape(1, -1)
        r = r[:, :self.d1] @ Q
        return r.detach()

    @staticmethod
    def print_info(train_acc, test_acc, attack_acc):
        print('our method')
        print_stat('train_acc', train_acc)
        print_stat('test_acc', test_acc)
        print_stat('attack_acc', attack_acc)
        print('')
