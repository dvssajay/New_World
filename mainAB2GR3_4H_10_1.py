'''Train CIFAR10 with PyTorch.'''
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
import wandb
import time 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

net = ab_2GR3_4H_10()

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ANDHRA_Bandersnatch2GR3_4H_10_Run1.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
                     
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Initialize the Weights & Biases run
wandb.init(project="New_World", config={
    "learning_rate": args.lr,
    "epochs": 200,
    "batch_size": 128,
    "model": "ab_2GR3_4H_10",
},     name="CIFAR10_ANDHRA_Bandersnatch2GR3_4H_1"  # Custom run name
)

# Record the start time
start_time = time.time()


# Training function
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Initialize counters for individual model accuracies
    correct_individual = [0] * 4
    total_individual = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out11, out12, out21, out22 = net(inputs)

        # Calculate losses for each output
        loss1 = criterion(out11, targets)
        loss2 = criterion(out12, targets)
        loss3 = criterion(out21, targets)
        loss4 = criterion(out22, targets)
        
        # Combine losses and backpropagate
        loss =  0.25 * (loss1 + loss2 + loss3 + loss4)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Predictions and majority voting
        outputs = [out11, out12, out21, out22]
        individual_predictions = [output.max(1)[1] for output in outputs]
        
        # Majority vote prediction
        p = torch.stack(individual_predictions, dim=0).cpu().detach().numpy()
        m = stats.mode(p)
        predicted_majority = torch.from_numpy(m[0]).squeeze().cuda()
        
        # Update majority correct count
        total += targets.size(0)
        correct += predicted_majority.eq(targets).sum().item()

        # Update individual model correct counts
        for i, pred in enumerate(individual_predictions):
            correct_individual[i] += pred.eq(targets).sum().item()
        total_individual += targets.size(0)

        # Display progress bar
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Log training loss and accuracy to wandb
    wandb.log({
        "train_loss": train_loss / len(trainloader), 
        "train_accuracy": 100. * correct / total,
        **{f"train_accuracy_model_{i+1}": 100. * correct_individual[i] / total_individual for i in range(4)}
    })


# Validation function
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Initialize counters for individual model accuracies
    correct_individual = [0] * 4
    total_individual = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out11, out12, out21, out22 = net(inputs)

            # Calculate losses for each output
            loss1 = criterion(out11, targets)
            loss2 = criterion(out12, targets)
            loss3 = criterion(out21, targets)
            loss4 = criterion(out22, targets)
        
            # Combine losses and backpropagate
            loss =  0.25 * (loss1 + loss2 + loss3 + loss4)

            test_loss += loss.item()
        
            # Predictions and majority voting
            outputs = [out11, out12, out21, out22]
            individual_predictions = [output.max(1)[1] for output in outputs]
            
            # Majority vote prediction
            p = torch.stack(individual_predictions, dim=0).cpu().detach().numpy()
            m = stats.mode(p)
            predicted_majority = torch.from_numpy(m[0]).squeeze().cuda()

            # Update majority correct count
            total += targets.size(0)
            correct += predicted_majority.eq(targets).sum().item()

            # Update individual model correct counts
            for i, pred in enumerate(individual_predictions):
                correct_individual[i] += pred.eq(targets).sum().item()
            total_individual += targets.size(0)

            # Display progress bar
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Log test loss and accuracy to wandb
    test_accuracy = 100. * correct / total
    wandb.log({
        "test_loss": test_loss / len(testloader), 
        "test_accuracy": test_accuracy,
        **{f"test_accuracy_model_{i+1}": 100. * correct_individual[i] / total_individual for i in range(4)}
    })


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ANDHRA_Bandersnatch2GR3_4H_10_Run1.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
    wandb.log({"epoch": epoch})


# Calculate total runtime
total_time = time.time() - start_time

# Log the total runtime to wandb (in seconds or formatted as hours:minutes:seconds)
wandb.log({"total_runtime_seconds": total_time})
wandb.log({"total_runtime_formatted": time.strftime("%H:%M:%S", time.gmtime(total_time))})


# End the wandb run
wandb.finish()