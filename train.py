import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import accuracy
from tqdm import tqdm

def prepare_dataset(train_X, train_Y, test_X, test_Y, batch_size):
    class TensorDataset(Dataset):
        def __init__(self, data_tensor, target_tensor):
            self.data_tensor = data_tensor
            self.target_tensor = target_tensor

        def __getitem__(self, index):
            return self.data_tensor[index], self.target_tensor[index]

        def __len__(self):
            return self.data_tensor.size(0)

    train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_Y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator())

    validation_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(train_Y))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, generator=torch.Generator())

    test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(test_Y))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, generator=torch.Generator())
    return train_dataset, train_loader, validation_dataset, validation_loader, test_dataset, test_loader

def train(net, data, verbose=False, use_gpu=False, skip=False):
    train_dataset, train_loader, validation_dataset, validation_loader = data
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    num_epoch = 100

    for epoch in range(1, num_epoch + 1):
        for i, (data, target) in enumerate(train_loader):
            if skip:
                continue
            optimizer.zero_grad()
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            #nn.utils.clip_grad_value_(net.parameters(), 2)
            '''if i % 100 == 0:
                print(net.input1.weight.detach().norm())
            if loss.cpu().item() > 100 and epoch > 1:
                print(loss.item())
                with torch.no_grad():
                    for j in range(data.shape[0]):
                        data_j = data[j, :].reshape(1, -1)
                        target_j = target[j].reshape(1, )
                        output_j = net.easy_forward(data_j)
                        loss_j = criterion(output_j, target_j)
                        if loss_j.item() > 300:
                            print('loss:', loss_j.item())
                            if loss_j.item() > 300:
                                print('data:', data_j)
                                print('target:', target_j)
                                print(output[j, :])
                                print('output', output_j)'''
            optimizer.step()
        if skip:
            continue
        scheduler.step()
        if verbose:
            with torch.no_grad():
                total_loss = 0.0
                total_acc = 0.0
                for i, (data, target) in enumerate(validation_loader):
                    if use_gpu:
                        data = data.cuda()
                        target = target.cuda()
                    output = net(data)
                    loss = criterion(output, target)
                    total_loss += loss.cpu().item() * len(data)
                    total_acc += accuracy(output, target).item() * len(data)
                total_loss /= len(validation_dataset)
                total_acc /= len(validation_dataset)
                print('epoch {} train loss:'.format(epoch), total_loss)
                print('epoch {} train acc:'.format(epoch), total_acc)
