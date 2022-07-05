import torch
import torch.nn as nn
import numpy as np
from utils import accuracy
from attack import equality_solve

def eval(net, data, binary_features):
    train_dataset, train_loader, test_dataset, test_loader = data
    criterion = nn.CrossEntropyLoss()
    train_acc = 0.0
    test_acc = 0.0
    A = []
    X = []
    # extract intermediate output
    def hook_forward_fn(module, input, output):
        A.append(output.numpy()[:, :net.d1])
    net.inter.register_forward_hook(hook_forward_fn)
    with torch.no_grad():
        for i, (data, target) in enumerate(train_loader):
            X.append(data.numpy())
            output = net(data)
            loss = criterion(output, target)
            train_acc += accuracy(output, target).item() * len(data)
        train_acc /= len(train_dataset)
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            X.append(data.numpy())
            output = net(data)
            loss = criterion(output, target)
            test_acc += accuracy(output, target).item() * len(data)
        test_acc /= len(test_dataset)
    A = np.concatenate(A, axis=0)
    X = np.concatenate(X, axis=0)
    ans = equality_solve(A)
    print('total {} solution(s).'.format(len(ans)))
    idx, best_acc = 0, 0
    for i in range(net.d1):
        for sol in ans:
            acc = np.sum(np.isclose(X[:, i], sol))
            if acc == X.shape[0]:
                print('attack feature no.{} successfully.'.format(i))
            acc /= X.shape[0]
            if acc > best_acc:
                idx, best_acc = i, acc
    for sol in ans:
        if np.sum(np.isclose(X[:, -1], sol)) == X.shape[0]:
            print('attack fake feature successfully.'.format(i))
    return train_acc, test_acc, best_acc, idx
