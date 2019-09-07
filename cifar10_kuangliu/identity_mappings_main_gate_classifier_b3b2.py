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

import argparse
import os
import time
from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter
import models.resnet_gate_classifier_b3b2 as models
import itertools
import numpy as np

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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
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
parser.add_argument('--scratch', default=1, type=int,
                    help='train from scratch or use the baseline to initialise part of te model')
parser.add_argument('--gate_iters', default=1, type=int,
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

if args.scratch==1:
    p = 'scratch'
else:
    p = 'bi'
print('check...........', p)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

print('\nrunning essential checks, ')
print('args.scratch {},\t args.gate_iters {},\t args.lr_gate {},\t args.schedule_gate {},\t args.schedule {},\t args.epochs {}'.format(args.scratch, args.gate_iters, args.lr_gate, args.schedule_gate, args.schedule, args.epochs))


def main():
    #if not os.path.exists(args.save_dir):
    #os.makedirs(args.save_dir)

    best_acc = 0  # best validation set accuracy
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..%s' % args.dataset)
    tf_dir = 'runs_c{}_{}_3248b2b3_{}_{}'.format(args.dataset[5:], args.arch+str(args.depth),p, args.mod_name_suffix)
    writer = SummaryWriter('cifar_resnets_modified/' + tf_dir)
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

    print('  Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # defining different optimizers for different losses backprop
    list_parameter = [
    model.module.conv1.parameters(),
    model.module.bn1.parameters(),
    model.module.layer1.parameters(),
    model.module.o21.parameters(),
    model.module.layer2.parameters(),
    model.module.o22.parameters(),
    model.module.o31.parameters(),
    model.module.layer3.parameters(),
    model.module.o32.parameters(),
    model.module.fc.parameters()]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(itertools.chain(*list_parameter), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_gate_b2 = torch.optim.SGD(model.module.g_layer_b2.parameters(), args.lr_gate,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    optimizer_gate_b3 = torch.optim.SGD(model.module.g_layer_b3.parameters(), args.lr_gate,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            new_dict_load = {}
            for k, v in checkpoint['state_dict'].items():
                new_dict_load['module.'+k] = v
            model.load_state_dict(new_dict_load)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            args.lr_gate = 0.001
            for param_group in optimizer_gate.param_groups:
                param_group['lr'] = args.lr_gate
            best_acc = checkpoint['acc']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format('yes', checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # to be modified specific to the layers of the resnet being used
    # this code is only for training from scratch, no baseline initialisation defined.

    if args.evaluate:
        print("Evaluation only\n")
        path = os.path.join('/data/chercheurs/garg191/checkpoint/cifar_resnets/', args.test_checkpoint)
        if os.path.isfile(path):
            print("=> loading test checkpoint")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
        acc, test_loss = validate(testloader, model, criterion)
        print("Test accuracy attained: {}, Test loss: {} ".format(acc, test_loss))
        return

    if args.depth in [1202, 110, 164]:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01


    test_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, optimizer_gate_b2, optimizer_gate_b3, epoch)
        if epoch == 1: # check for 2nd epoch
            # after the 1st epoch (0th), bring it back to initial lr = 0.1 and let the normal lr schedule follow
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs - 1, optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(trainloader, model, criterion, optimizer, optimizer_gate_b2, optimizer_gate_b3, epoch)

        # evaluate on validation set
        test_accuracy, test_loss, test_loss_cls, test_loss_gatecel = validate(testloader, model, criterion, 0)
        train_accuracy, train_loss, train_loss_cls, train_loss_gatecel = validate(trainloader, model, criterion, 1)

        info = {'tot_test_loss': test_loss, 'test_loss_gatecel': test_loss_gatecel, 'test_loss_cls': test_loss_cls, 'test_accuracy': test_accuracy, 'tot_train_loss': train_loss, 'train_loss_cls': train_loss_cls, 'train_loss_gatecel': train_loss_gatecel, 'train_accuracy': train_accuracy, 'lr': optimizer.param_groups[0]['lr']}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch+1)

        # remember best prec@1 and save checkpoint
        is_best = test_accuracy > best_acc
        best_acc = max(test_accuracy, best_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': best_acc,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_best.pth')

        if epoch % 50 == 0 and epoch!=0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': test_accuracy,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_{}.pth'.format(epoch))


    print("final test accuracy at the end of {} epochs: ".format(args.epochs), test_accuracy)
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': test_accuracy,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_final_{}.pth'.format(epoch))



def train(trainloader, model, criterion, optimizer, optimizer_gate_b2, optimizer_gate_b3, epoch):
    """
        Run one train epoch
    """
    model.train()
    correct = 0
    total = 0
    tot_loss = 0
    flag = 0
    for ind, (inputs, target) in enumerate(trainloader):
        # measure data loading time
        inputs = inputs.to(device)
        target = target.to(device)

        #print("\nTest 1, g_layer_b2 weights before gate loss gradient update ", model.module.g_layer_b2[0].weight.mean().item())
        #print("\nTest 1, g_layer_b3 weights before gate loss gradient update ", model.module.g_layer_b3[0].weight.mean().item())
        #print("\nTest 1, layer 2 weights before gate loss gradient update ", model.module.layer1[0].conv2.weight.mean().item())

        # compute the gate classifier loss and backprop it through only the gate FC block defined by g_layer
        model_output, gate_b2_out, gate_b3_out = model(inputs)
        gate_cel_b2 = criterion(gate_b2_out, target)
        gate_cel_b3 = criterion(gate_b3_out, target)

        optimizer_gate_b2.zero_grad()
        gate_cel_b2.backward(retain_graph=True)
        optimizer_gate_b2.step()

        optimizer_gate_b3.zero_grad()
        gate_cel_b3.backward(retain_graph=True)
        optimizer_gate_b3.step()

        #print("\nTest 2, g_layer_b2 weights after gate loss gradient update ", model.module.g_layer_b2[0].weight.mean().item())
        #print("\nTest 2, g_layer_b3 weights after gate loss gradient update ", model.module.g_layer_b3[0].weight.mean().item())
        #print("\nTest 2, layer 2 weights after gate loss gradient update ", model.module.layer1[0].conv2.weight.mean().item())

        # backprop classification loss only when ind%args.gate_iters == 0
        if (ind+1) % args.gate_iters == 0:
            # calculate the main classification loss using the output of the network
            #model_output, gate_block_out = model(inputs)
            classification_loss = criterion(model_output, target)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            #print("\nTest 3, g_layer_b2 weights after classification loss gradient update ", model.module.g_layer_b2[0].weight.mean().item())
            #print("\nTest 3, g_layer_b3 weights after classification loss gradient update ", model.module.g_layer_b3[0].weight.mean().item())
            #print("\nTest 3, layer 2 weights after classification loss gradient update ", model.module.layer1[0].conv2.weight.mean().item())

            tot_loss += classification_loss.item()

        _, predicted = model_output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        progress_bar(ind, len(trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (tot_loss/(ind+1), 100.*correct/total, correct, total))


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
        for i, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.to(device), target.to(device)
            output, gate_out_b2, gate_out_b3 = model(inputs)
            loss_gate_b2 = criterion(gate_out_b2, target)
            loss_gate_b3 = criterion(gate_out_b3, target)
            classification_loss = criterion(output, target)
            loss = loss_gate_b2 + loss_gate_b3 + classification_loss

            test_loss += loss.item()
            test_loss_cls += classification_loss.item()
            test_loss_gatecel += loss_gate_b2.item() + loss_gate_b3.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if not flag:
                progress_bar(i, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(i+1), 100.*correct/total, correct, total))

    # losses logged are - total loss, classification only loss, sum of both gate losses
    acc = 100.*correct/total
    return acc, test_loss/total, test_loss_cls/total, test_loss_gatecel/total


def adjust_learning_rate(optimizer, optimizer_gate_b2, optimizer_gate_b3, epoch):
    """Sets the learning rate to the present LR decayed by lr_decay_gamma at every point in the schedule"""
    if epoch == args.schedule[0]:
        lr = args.lr*(args.lr_decay_gamma**1)
    elif epoch == args.schedule[1]:
        lr = args.lr*(args.lr_decay_gamma**2)
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if args.schedule_gate!=0 and (epoch+1) % 50 == 0:
        lr_gate = args.lr_gate * (0.1 ** ((epoch+1) // 50))
        for param_group in optimizer_gate_b2.param_groups:
            param_group['lr'] = lr_gate
        for param_group in optimizer_gate_b3.param_groups:
            param_group['lr'] = lr_gate

def save_checkpoint(state, name):
    """
    Save the training model
    """
    path = '/data/chercheurs/garg191/checkpoint/cifar_resnets_modified/' + p + '/c{}_{}_3248b2b3_{}'.format(args.dataset[5:], args.arch+str(args.depth), args.mod_name_suffix)
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))
    print("checkpoint saved: ", name)


if __name__ == '__main__':
    main()
