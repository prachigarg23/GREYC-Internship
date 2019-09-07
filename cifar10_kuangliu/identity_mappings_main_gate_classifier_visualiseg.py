# final code for cifar training
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import *
import csv
import pandas as pd

import argparse
import os
import time
from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter
import models.resnet_gate_classifier_visualiseg as models
import itertools
import numpy as np
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar 10 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[120, 180],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='baseline_models', type=str)
parser.add_argument('--lr_decay_step', default=50, type=int, help='step size after which learning rate is decayed')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float,
                    help='gamma value for lr decay')
parser.add_argument('--test_checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--s_no', default=1, type=int,
                    help='to track which out of the 5 trained models of the same model it is')
parser.add_argument('--lambda_mem', default=0.5, type=float,
                    help='memorization loss hyperparameter')
parser.add_argument('--scratch', default=0, type=int,
                    help='train from scratch or use the baseline to initialise part of te model')
parser.add_argument('--gate_iters', default=3, type=int,
                    help='the number of times the gate classifer is to be trained')
parser.add_argument('--lr_gate', '--learning-rate_gate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate for the gate FC block')
parser.add_argument('--schedule_gate', default=0, type=int,
                    help='0 if want to keep gate lr constant, 1 for reducing it by 0.1 every 50 epochs')
parser.add_argument('--mod_name_suffix', default='', type=str,
                    help='name to make model easily identifiable')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("TEST if cuda is working or not: ", device)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

print('\nThis file is to get and visualise the value of g for each sample in the dataset passed (train or test)\n')
if args.scratch==1:
    p = 'scratch'
else:
    p = 'bi'
print('check...........', p)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

def main():
    #if not os.path.exists(args.save_dir):
    #os.makedirs(args.save_dir)

    best_acc = 0  # best validation set accuracy
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..%s' % args.dataset)
    #tf_dir = 'runs_c{}_{}_3248_{}_{}_{}'.format(args.dataset[5:], args.arch+str(args.depth),p, args.mod_name_suffix, args.s_no)
    #writer = SummaryWriter('cifar_resnets_modified/' + tf_dir)
    transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),])
    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),])

    if args.dataset == 'cifar10':
        dataloader = torchvision.datasets.CIFAR10
        num_classes = 10
        data_dir = '/home/personnels/garg191/cifar10_kuangliu/data'
        baseline_dir = '/home/personnels/garg191/baselines/c10_resnet110_3248/checkpoint_final.pth'
    else:
        dataloader = torchvision.datasets.CIFAR100
        num_classes = 100
        data_dir = '/home/personnels/garg191/cifar100/data'
        baseline_dir = '/home/personnels/garg191/baselines/c100_resnet164_3248/checkpoint_final.pth'

    trainset = dataloader(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes = num_classes,
                    depth = args.depth,
                    block_name = args.block_name,
                    )

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        print("Evaluation only\n")
        path = ''
        #path = os.path.join('/data/chercheurs/garg191/checkpoint/cifar_resnets_modified/scratch/c10_resnet_gate_classifier110_3248_it1g-2cm-1', args.test_checkpoint)
        if os.path.isfile(path):
            print("=> loading test checkpoint")
            checkpoint = torch.load(path)
            new_dict_to_load = {}
            print('\ncheckpoint keys\n ')
            for k, v in checkpoint['state_dict'].items():
                new_dict_to_load['module.'+k] = v
            model.load_state_dict(new_dict_to_load, strict=True)
        acc, test_loss, test_loss_cls, test_loss_gatecel = validate(testloader, model, criterion)
        print("Train accuracy attained: {}, Train loss: {} ".format(acc, test_loss))
        return

def validate(testloader, model, criterion, flag=0):
    """
    Run evaluation
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_gatecel = 0.0
    test_loss_cls = 0.0
    # at test time, there's a single loss which is the classification loss.
    # tho its possible to use the g_layer_out here to get the gate classifer loss, but technically it doesn't make sense cuz it has no role to play at test time.
    with torch.no_grad():
        g_vector_hist = []
        targets_hist = []
        predicted_hist = []
        for i, (inputs, target) in enumerate(testloader):
            targets_hist.append(target[0].numpy())
            inputs, target = inputs.to(device), target.to(device)
            output, gate_out, g_value = model(inputs)
            # the g_value is the normalised value of g for each example,
            # after the original g (entropy) has been centred around the origin and passed through sigmoid
            # Hence, this g_value has to be in the range [0,1]
            if args.batch_size != 1:
                print('please use batch size 1 ')
                return

            g_vector_hist.append(g_value[0].cpu().numpy())
            loss_gate = criterion(gate_out, target)
            classification_loss = criterion(output, target)
            loss = loss_gate + classification_loss

            test_loss += loss.item()
            test_loss_cls += classification_loss.item()
            test_loss_gatecel += loss_gate.item()
            _, predicted = output.max(1)
            predicted_hist.append(predicted[0].cpu().numpy())

            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            #if not flag:
            #    progress_bar(i, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #        % (test_loss/(i+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
        df = pd.DataFrame({"g value" : g_vector_hist, "True labels" : targets_hist, "Predicted labels" : predicted_hist})
        df.to_csv("g_c10_trainset_after300epochs.csv", index=True)
        print("written")
        #plot_hist(hist_matrix)
        return acc, test_loss/total, test_loss_cls/total, test_loss_gatecel/total


def plot_hist(hist_matrix):
    hist_matrix = np.array(hist_matrix)
    print('size of hist_matrix: ', hist_matrix.shape, type(hist_matrix))
    print(hist_matrix)
    print('distribution of values: ', np.histogram(hist_matrix))
    print('max, min ', hist_matrix.max(), hist_matrix.min())
    fig = plt.hist(hist_matrix, bins=10, rwidth=0.85)
    plt.title('Gate classifier, Model 4, scratch_it1g-2cm-1 trainset, G VALUE HISTOGRAM')
    plt.xlabel('g value')
    plt.ylabel('Frequency')
    plt.savefig('scit1g-2cm-1_hist_10bins_trainset.png')

if __name__ == '__main__':
    main()
