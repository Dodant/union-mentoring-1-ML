import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import time


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


def after_amp(pin_memory, num_workers):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5)])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    batch_size = 1024


    cifarset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainloader = DataLoader(cifarset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scaler = amp.GradScaler()

    epochs = 30

    train_st = time.time()
    for epoch in range(epochs):
        # preprocessing time
        p_list = []
        for inputs, labels in trainloader:
            p_list.append(time.time())
            
            # dataloading time
            l_st = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            l_end = time.time()
            
            # forward time
            f_st = time.time()
            with amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            f_end = time.time()
            
            # backward time
            b_st = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            b_end = time.time()
        
        p_time = p_list[-1] - p_list[-2]
        l_time = l_end - l_st
        f_time = f_end - f_st
        b_time = b_end - b_st
        T_time = p_time + l_time + f_time + b_time

        if epoch == epochs-1:
            print(f'Config: pm={pin_memory}, nw={num_workers} || preprocess = {p_time:0.4f}s, load = {l_time:0.4f}s, forward = {f_time:0.4f}s, backward = {b_time:0.4f}s')
            print(f'Pre | Load | FW | BW : {p_time*100/T_time:0.3f}% | {l_time*100/T_time:0.3f}% | {f_time*100/T_time:0.3f}% | {b_time*100/T_time:0.3f}%')
    
    train_end = time.time()
    print(f'Total Training Time : {train_end-train_st:0.1f}s')    
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy : {correct*100/total:0.1f}%')
    print('===========================================')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p')
    parser.add_argument('-n', type=int)
    
    args = parser.parse_args()
    if args.p == 'True':
        after_amp(True, args.n)
    elif args.p == 'False':
        after_amp(False, args.n)
    print("Done")