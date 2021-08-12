import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from functools import reduce
import numpy as np
import argparse
import math
import json

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((64, 64)),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

batch_size = 1024


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


def prune_and_test(p):

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)


    net = Net()
    PATH = './cifar_net.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    total_W = 0
    total_Z = 0

    layers = ['layer1.0.weight','layer1.3.weight',
                'layer2.0.weight','layer2.3.weight',
                'layer3.0.weight','layer3.3.weight',
                'fc1.weight','fc2.weight','fc3.weight','fc4.weight']
    print('p =', p)
    
    for layer in layers:
        print(layer)
        target = net.state_dict()[layer].data
        flatten = target.view(-1)
        boundary = abs(sorted(flatten, key=lambda a: torch.abs(a))[math.ceil(len(flatten) * p / 100)].item())
        lower = -boundary < target
        upper = target < boundary
        
        target = torch.where(torch.logical_not(torch.logical_and(lower, upper)), target, torch.cuda.FloatTensor([0]))
        net.state_dict()[layer].data.copy_(target)
        
        total_Z += len(net.state_dict()[layer][net.state_dict()[layer].data == 0.0])
        total_W += reduce(lambda x,y: x*y, net.state_dict()[layer].size())


    train_correct, train_total = 0, 0
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
    data = {}
    data['result'] = []
    data['result'].append({
        'p': p,
        'trainset_acc': 100 * train_correct / train_total,
        'testset_acc': 100 * test_correct / test_total,
        'num_of_zeros': total_Z,
        'num_of_weights': total_W,
        'pruned_ratio': total_Z / total_W * 100
    })

    print(f'p = {p / 100}')
    print(f'Trainset Acc: {100 * train_correct / train_total: .1f}%')
    print(f'Testset Acc: {100 * test_correct / test_total: .1f}%')
    print(f'Number of Zeros: {total_Z}')
    print(f'Number of Weights: {total_W}')
    print(f'Pruned Ratio: {total_Z / total_W * 100: .1f}%')


    file_path = './' + (str(p) if len(str(p)) != 1 else '0' + str(p)) + '.json'
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

    PATH = './cifar_prune_' + str(p) + '.pth'
    torch.save(net.state_dict(), PATH)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int)
    
    args = parser.parse_args()
    prune_and_test(args.p)
    print("Done")