import torch
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import torchvision
from torchvision import models, transforms, datasets

import time
import argparse

def DistillationLoss(student_logit, teacher_logit, T):
    soft_label = F.softmax(teacher_logit/T, dim=1)
    soft_prediction = F.log_softmax(student_logit/T, dim=1)
    return F.kl_div(soft_prediction, soft_label)

def FinalLoss(teacher_logit, student_logit, labels, T, alpha):
    return (1.-alpha)*F.cross_entropy(student_logit, labels) + (alpha*T*T)*DistillationLoss(student_logit, teacher_logit, T)

def kd_train(student_model, T):
    # hyperparameter
    batch_size = 256
    num_workers = 4

    epochs = 25
    student = 0
    alpha = 0.7

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.244, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5)
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # teacher = ResNet50
    teacher = models.resnet50(pretrained=True).to(device).eval()

    # student = ResNet18, ResNet34, ResNet50
    if student_model == 18:
        student = models.resnet18(pretrained=False).to(device)
    elif student_model == 34:
        student = models.resnet34(pretrained=False).to(device)
    elif student_model == 50:
        student = models.resnet50(pretrained=False).to(device)
        

    imagenet = datasets.ImageFolder('./imagenet-object-localization-challenge/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train', transform=transform)
    trainset, testset = random_split(imagenet, [1_000_000, len(imagenet) - 1_000_000]) 
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scaler = amp.GradScaler()

    print(f'Config : Student = ResNet{student_model}, T = {T}, alpha = {alpha}')


    train_st = time.time()
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with amp.autocast():
                student_logit = student(inputs)
                teacher_logit = teacher(inputs)
                loss = FinalLoss(teacher_logit, student_logit, labels, T, alpha)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(student_logit, 1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data)
            train_samples += len(inputs)
        
        epoch_loss = train_loss / len(trainloader)
        epoch_acc = train_acc.float() / train_samples * 100
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch: {epoch + 1} || tl: {epoch_loss:.3f}, ta: {epoch_acc:.2f}%")

    train_end = time.time()
    print(f'Training Finished - Train time : {(train_end - train_st)//60}m\n')
    
    PATH = './student_' + str(student_model) + '_' + str(T) + '.pth'
    torch.save(student.state_dict(), PATH)


    correct_s, correct_5s, total_s = 0, 0, 0
    correct_t, correct_5t, total_t = 0, 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # student - top1
            outputs_s = student(images)
            _, predicted_s = torch.max(outputs_s.data, 1)
            total_s += labels.size(0)
            correct_s += (predicted_s == labels).sum().item()
            # top5
            for idx, item in enumerate(labels.view(-1,1)):
                if item in torch.topk(outputs_s, 5).indices[idx]:
                    correct_5s += 1
            
            # teacher - top1
            outputs_t = teacher(images)
            _, predicted_t = torch.max(outputs_t.data, 1)
            total_t += labels.size(0)
            correct_t += (predicted_t == labels).sum().item()
            # top5
            for idx, item in enumerate(labels.view(-1,1)):
                if item in torch.topk(outputs_t, 5).indices[idx]:
                    correct_5t += 1

    print(f'Top1 Acc : student - {correct_s*100/total_s:0.1f}% / teacher - {correct_t*100/total_t:0.1f}%')
    print(f'Top5 Acc : student - {correct_5s*100/total_s:0.1f}% / teacher - {correct_5t*100/total_t:0.1f}%')
    print('===========================================\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int)
    parser.add_argument('-t', type=int)
    
    args = parser.parse_args()
    kd_train(args.s, args.t)