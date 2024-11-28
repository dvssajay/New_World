'''Test CIFAR100 with 8 heads using ensemble prediction methods'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from scipy import stats
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar
import sys

# Argument Parsing
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Testing')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data Preparation
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

testset = torchvision.datasets.CIFAR100(root='./data100', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Model Setup
print('==> Building model..')
net = ab_2GR3_100()  # Replace with actual network
net = net.to(device)

# Load Checkpoint
print('==> Loading from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ANDHRA_Bandersnatch2GR3_100_Run1.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

# Track Top-Performing Head
top_head_num = -1
top_head_acc = 0

def test_head(num):
    global top_head_num, top_head_acc
    net.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)  # Obtain outputs from all heads
            loss = criterion(outputs[num], targets)
            test_loss += loss.item()

            # Top-1 Accuracy
            _, predicted = outputs[num].max(1)
            correct_top1 += predicted.eq(targets).sum().item()

            # Top-5 Accuracy
            _, top5_pred = outputs[num].topk(5, dim=1)
            correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

            total += targets.size(0)

            progress_bar(batch_idx, len(testloader), 
                         f'Head-{num+1} | Loss: {test_loss/(batch_idx+1):.3f} | Top-1 Acc: {100.*correct_top1/total:.3f}%')

    print(f"\nHead-{num+1} Test Summary:")
    print(f"Top-1 Acc: {100.*correct_top1/total:.3f}% ({correct_top1}/{total})")
    print(f"Top-5 Acc: {100.*correct_top5/total:.3f}% ({correct_top5}/{total})\n")

    # Update top-performing head
    if 100. * correct_top1 / total > top_head_acc:
        top_head_acc = 100. * correct_top1 / total
        top_head_num = num + 1  # Head number



def test_combined():
    net.eval()
    test_loss = 0
    correct_counts = defaultdict(int)
    total = 0

    # Track individual head performance
    individual_heads_correct = [0] * 8  # Adjust for number of heads in your network
    top1_scores = defaultdict(int)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            # Calculate loss (optional)
            loss = sum(criterion(out, targets) for out in outputs) / len(outputs)
            test_loss += loss.item()

            # Collect individual head predictions
            predictions = [out.max(1)[1] for out in outputs]
            probabilities = [F.softmax(out, dim=1) for out in outputs]

            # Majority Voting
            stacked_preds = torch.stack(predictions, dim=0)
            majority_vote = stats.mode(stacked_preds.cpu().numpy(), axis=0).mode.squeeze()
            majority_vote = torch.tensor(majority_vote).to(device)
            correct_counts['majority'] += majority_vote.eq(targets).sum().item()

            # Weighted Voting
            head_weights = [correct / (total + 1e-5) for correct in individual_heads_correct]
            total_weight = sum(head_weights)
            if total_weight == 0:  # Avoid division by zero
                normalized_weights = [1.0 / len(head_weights)] * len(head_weights)
            else:
                normalized_weights = [w / total_weight for w in head_weights]

            weighted_votes = sum(w * F.one_hot(pred, num_classes=100).float() for w, pred in zip(normalized_weights, predictions))
            weighted_vote = weighted_votes.argmax(1)
            correct_counts['weighted'] += weighted_vote.eq(targets).sum().item()

            # Average Probability
            avg_probs = torch.stack(probabilities, dim=0).mean(0)
            avg_prob_pred = avg_probs.argmax(1)
            correct_counts['average_prob'] += avg_prob_pred.eq(targets).sum().item()

            # Product of Experts (PoE)
            product_probs = torch.exp(torch.log(torch.stack(probabilities, dim=0) + 1e-10).sum(0))
            poe_pred = product_probs.argmax(1)
            correct_counts['poe'] += poe_pred.eq(targets).sum().item()

            # Rank-Based Voting
            rank_scores = torch.zeros((inputs.size(0), 100), dtype=torch.float32, device=device)
            for prob in probabilities:
                _, rank = prob.sort(descending=True, dim=1)
                for idx in range(rank.size(1)):
                    rank_scores.scatter_add_(
                        1,
                        rank[:, idx:idx + 1],
                        torch.full_like(rank[:, idx:idx + 1], 1.0 / (idx + 1), dtype=torch.float32)
                    )
            rank_vote = rank_scores.argmax(1)
            correct_counts['rank_based'] += rank_vote.eq(targets).sum().item()

            total += targets.size(0)
            progress_bar(
                batch_idx, len(testloader),
                f"Combined | Acc: {100.*correct_counts['majority']/total:.2f}%"
            )

        # Log all accuracies
        for method, correct in correct_counts.items():
            top1_scores[method] = 100. * correct / total

        print(f"Combined Accuracies: {dict(top1_scores)}")



# Test Combined
print("\nTesting Combined Model:")
test_combined()



# Test All Heads Individually
for i in range(8):
    print(f"\nTesting Head-{i+1}:")
    test_head(i)

# Report Top-Performing Head
print(f"\nTop-Performing Head: Head-{top_head_num} with Top-1 Acc: {top_head_acc:.3f}%\n")

