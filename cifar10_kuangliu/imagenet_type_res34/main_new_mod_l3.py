# final code for cifar 100 training
# change the batch size argument in dataloaders
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import argparse
import os
import time
from utils import progress_bar
from resnet import *
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Cifar 10 Training')
#parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
#                    choices=model_names)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
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
                    default='/data/chercheurs/garg191/', type=str)
parser.add_argument('--lr_decay_step', default=50, type=int, help='step size after which learning rate is decayed')
parser.add_argument('--lr_decay_gamma', default=0.1, type=float,
                    help='gamma value for lr decay')
parser.add_argument('--test_checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none), give dir of model as well as name of checkpoint')
parser.add_argument('--lambda_mem', default=0.5, type=float,
                    help='memorization loss hyperparameter')
parser.add_argument('--scratch', default=True, type=bool,
                    help='train from scratch or use the baseline to initialise part of the model')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("TEST if cuda is working or not: ", device)
args = parser.parse_args()
# x defines the name of the runs directory and checkpoints save directory specific to each model
x = ''
l = args.lambda_mem
if l <= 0.00001:
    x = str(l)[0] + 'e-5'
elif l <= 0.0001:
    x = str(l)[0] + 'e-4'
elif l <= 0.001:
    x = str(l)[0] + 'e-3'
elif l <= 0.01:
    x = str(l)[0] + 'e-2'
elif l <= 0.1:
    x = str(l)[0] + 'e-1'
elif l <= 1.0:
    x = str(l)[0] + 'e+0'
elif l <= 10.0:
    x = str(l)[0] + 'e+1'
elif l <= 100.0:
    x = str(l)[0] + 'e+2'
elif l <= 1000.0:
    x = str(l)[0] + 'e+3'

if args.scratch==True:
    p = 'scratch'
else:
    p = 'baseline_initialisation'


def main():
    #if not os.path.exists(args.save_dir):
    #os.makedirs(args.save_dir)

    best_acc = 0  # best validation set accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print("scratch::::::::::", p)
    print('==> Preparing data..')
    path = 'runs_c10_res34_3248_/' + p + x
    print("saving logs to:", path)
    writer = SummaryWriter(path)
    transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),])
    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("BUILDING MODEL...")

    # defining the model as inbuilt model
    model = ResNet34()
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # if args.scratch is false, then load the baseline model
    if args.scratch == False:
        ckpt = torch.load('/data/chercheurs/garg191/checkpoint/baseline_c10_res34_3248/checkpoint_final.pth')
        print("loaded checkpoint\n")
        new_dict_to_load = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.layer1') or k.startswith('module.layer2') or k.startswith('module.layer4') or k.startswith('module.conv') or k.startswith('module.bn'):
                new_dict_to_load[k] = v
            if k.startswith('module.layer3.0.'):
                new_key = k.replace('module.layer3.0.', 'module.o1.')
                new_dict_to_load[new_key] = v
            if k.startswith('module.layer3.1.'):
                new_key = k.replace('module.layer3.1.', 'module.layer3.0.')
                new_dict_to_load[new_key] = v
            if k.startswith('module.layer3.2.'):
                new_key = k.replace('module.layer3.2.', 'module.layer3.1.')
                new_dict_to_load[new_key] = v
            if k.startswith('module.layer3.3.'):
                new_key = k.replace('module.layer3.3.', 'module.layer3.2.')
                new_dict_to_load[new_key] = v
            if k.startswith('module.layer3.4.'):
                new_key = k.replace('module.layer3.4.', 'module.layer3.3.')
                new_dict_to_load[new_key] = v
            if k.startswith('module.layer3.5.'):
                new_key = k.replace('module.layer3.5.', 'module.o2.')
                new_dict_to_load[new_key] = v

        model.load_state_dict(new_dict_to_load, strict=False)

    if args.evaluate:
        if os.path.isfile(args.test_checkpoint):
            print("=> loading test checkpoint")
            checkpoint = torch.load(args.test_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
        acc, test_loss = validate(testloader, model, criterion)
        print("Test accuracy attained: {}, Test loss: {} ".format(acc, test_loss))
        return

    test_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % 82 == 0 or epoch % 123 == 0:
            adjust_learning_rate(optimizer, epoch)
        print("Epoch: ", epoch)
        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        test_accuracy, test_loss = validate(testloader, model, criterion, 0)
        train_accuracy, train_loss = validate(trainloader, model, criterion, 1)

        info = { 'test_loss': test_loss, 'test_accuracy': test_accuracy, 'train_loss': train_loss, 'train_accuracy': train_accuracy }
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

    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'acc': test_accuracy,
                'optimizer': optimizer.state_dict(),
                }, 'checkpoint_final.pth')



def train(trainloader, model, criterion, optimizer, epoch):
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

        # compute output
        output, g = model(inputs)
        classification_loss = criterion(output, target)
        l2_norm = torch.mean(g**2, 0)
        if flag == 0 and epoch % 10 == 0:
            print("classification loss: {}, l2_norm: {}".format(classification_loss, l2_norm))
            flag = 1
        mem_loss = args.lambda_mem * l2_norm
        loss = classification_loss + mem_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        progress_bar(ind, len(trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (tot_loss/(ind+1), 100.*correct/total, correct, total))


def validate(testloader, model, criterion, flag=0):
    """
    Run evaluation
    """
    model.eval()
    #end = time.time()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.to(device), target.to(device)
            output, _ = model(inputs)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if not flag:
                progress_bar(i, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(i+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    return acc, test_loss

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by lr_decay_gamma every lr_decay_step epochs"""
#    lr = args.lr * (args.lr_decay_gamma ** (epoch // args.lr_decay_step))
    lr = 0.0
    if epoch == 82:
        lr = args.lr * (args.lr_decay_gamma ** 1)
    elif epoch == 123:
        lr = args.lr * (args.lr_decay_gamma ** 2)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, name):
    """
    Save the training model
    """
    path = '/data/chercheurs/garg191/checkpoint/mod_l3_res34_3248/' + p + '/ml3_gl2_{}'.format(x)
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))
    print("checkpoint saved: ", path+name)


if __name__ == '__main__':
    main()
