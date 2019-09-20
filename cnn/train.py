#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:19:34
#
######
from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from dataset import Infrared_Dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv1 = nn.Conv2d(1, 20, 3, 2)
        self.conv2 = nn.Conv2d(20, 50, 3, 2)
        self.conv3 = nn.Conv2d(50, 80, 3, 2)
        self.conv4 = nn.Conv2d(80,30,3,2)
        # self.fc1 = nn.Linear(4*4*50, 500)
        self.fc1 = nn.Linear(30, 500)
        self.fc2 = nn.Linear(500, 7)

    def forward(self, x):
        # print(x.size(0))
        # x = torch.reshape(x,[1,1,328,338])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--base_dir')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #train_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST('../data', train=True, download=True,
    #                   transform=transforms.Compose([
    #                       transforms.ToTensor(),
    #                       transforms.Normalize((0.1307,), (0.3081,))
    #                   ])),
    #    batch_size=args.batch_size, shuffle=True, **kwargs)
    #test_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                       transforms.ToTensor(),
    #                       transforms.Normalize((0.1307,), (0.3081,))
    #                   ])),
    #    batch_size=args.test_batch_size, shuffle=True, **kwargs)
    training_file = os.path.join(args.base_dir, "train.lst")
    test_file = os.path.join(args.base_dir, "test.lst")
    train_loader = torch.utils.data.DataLoader(
                       Infrared_Dataloader(training_file, test_file, label_dim=8,
                           train=True,transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.12405,), (0.03575,))
                       ])),
          batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
                        Infrared_Dataloader(training_file, test_file, label_dim=8,
                           train=False,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.12405,), (0.03575,))
                       ])),
          batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        print("t1:",time.time()-t1)
        t2 = time.time()
        test(args, model, device, test_loader)
        print("t2:", time.time() - t2)
    if (args.save_model):
        torch.save(model.state_dict(),"5f2d.pt")
        
if __name__ == '__main__':
    main()
