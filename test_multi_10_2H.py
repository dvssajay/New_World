'''Test CIFAR10 with 2 Heads'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from scipy import stats
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar
import sys


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

net = ab_2GR3_2H3_10()

net = net.to(device)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

# Load checkpoint.
print('==> Loading from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ANDHRA_Bandersnatch2GR3_2H3_10_Run1.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


def test(num):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out1, out2 = net(inputs)
            loss1 = criterion(out1, targets)
            loss2 = criterion(out2, targets)
            
            loss =  0.5 * (loss1 + loss2)     

            test_loss += loss.item()
            
            out1, out2  = out1.max(1), out2.max(1)  
        
            _, predicted1 = out1
            _, predicted2 = out2 
          
            predictions = torch.stack((predicted1, predicted2),dim=0)

            total += targets.size(0)
            correct += predictions[num].eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))                      


net.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        out1, out2 = net(inputs)
        loss1 = criterion(out1, targets)
        loss2 = criterion(out2, targets)
            
        loss =  0.5 * (loss1 + loss2) 

        test_loss += loss.item()
            
        out1, out2  = out1.max(1), out2.max(1)   
  
        _, predicted1 = out1
        _, predicted2 = out2
            
        predictions = torch.stack((predicted1, predicted2),dim=0)
        
        p = torch.stack((predicted1, predicted2),dim=0).cpu().detach().numpy()


        m = stats.mode(p)   
       
        predicted = torch.from_numpy(m[0]).cuda()
    
        total += targets.size(0)
        
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        


for i in range(2):
    print('model')
    print(i)
    test(i)




            













