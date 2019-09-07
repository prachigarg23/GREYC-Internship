# final code for cifar training
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
from torch.utils.tensorboard import SummaryWriter
import models as models


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
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[82, 123],
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
parser.add_argument('--lambda_mem', default=0.1, type=float,
                    help='memorization loss hyperparameter')
parser.add_argument('--scratch', default=0, type=int,
                    help='train from scratch or use the baseline to initialise part of te model')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("TEST if cuda is working or not: ", device)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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

if args.scratch==1:
    p = 'scratch'
else:
    p = 'bi'
print('check.............', p)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

def main():
    #if not os.path.exists(args.save_dir):
    #os.makedirs(args.save_dir)

    best_acc = 0  # best validation set accuracy
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..%s' % args.dataset)
    tf_dir = 'runs_c{}_{}_3248_{}_cel'.format(args.dataset[5:], args.arch+str(args.depth),p+x)
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

    print("BUILDING MODEL...'{}'".format(args.arch))

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
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    criterion_gate = nn.CrossEntropyLoss()
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

    # to be modified specific to the layers of the resnet being used
    # if args.scratch is false, then load the baseline model
    if args.scratch == 0:
        if args.block_name.lower() == 'basicblock':
            n = (args.depth - 2) // 6
        elif args.block_name.lower() == 'bottleneck':
            n = (args.depth - 2) // 9

        ckpt = torch.load(baseline_dir)
        print("loaded checkpoint\n", baseline_dir)
        new_dict_to_load = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.layer1') or k.startswith('module.layer2') or k.startswith('module.conv') or k.startswith('module.bn'):
                new_dict_to_load[k] = v
            elif k.startswith('module.layer3.0.'):
                new_key = k.replace('module.layer3.0.', 'module.o1.')
                new_dict_to_load[new_key] = v
            elif k.startswith('module.layer3.{}.'.format(n-1)):
                new_key = k.replace('module.layer3.{}.'.format(n-1), 'module.o2.')
                new_dict_to_load[new_key] = v
            elif k.startswith('module.layer3.'):
                int_ = int(k[14])
                new_key = k
                new_key[14] = int_ - 1
                new_dict_to_load[new_key] = v

        model.load_state_dict(new_dict_to_load, strict=False)


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
        adjust_learning_rate(optimizer, epoch)
        if epoch == 1: # check for 2nd epoch
            # after the 1st epoch (0th), bring it back to initial lr = 0.1 and let the normal lr schedule follow
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs - 1, optimizer.param_groups[0]['lr']))

        # train for one epoch
        train(trainloader, model, criterion,  criterion_gate, optimizer, epoch)

        # evaluate on validation set
        test_accuracy, test_loss, test_loss_cls, test_loss_cel = validate(testloader, model, criterion, criterion_gate, 0)
        train_accuracy, train_loss, train_loss_cls, train_loss_cel = validate(trainloader, model, criterion, criterion_gate, 1)

        info = { 'tot_test_loss': test_loss, 'test_loss_cls': test_loss_cls, 'test_loss_cel': test_loss_cel, 'test_accuracy': test_accuracy, 'tot_train_loss': train_loss, 'train_loss_cls': train_loss_cls, 'train_loss_cel': train_loss_cel, 'train_accuracy': train_accuracy}
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



def train(trainloader, model, criterion, criterion_gate, optimizer, epoch):
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

        # calculating the cross entropy loss for the gate
        r = g.size(0)
        #gate_target = torch.ones([r, 2], dtype=torch.float32)
        #gate_target[:,1] = 0.0
        gate_target = torch.zeros([r], dtype=torch.long)
        gate_target = gate_target.to(device)
        g = g.to(device)
        gate_cel = criterion_gate(g, gate_target)

        if flag == 0 and epoch % 10 == 0:
            print("classification loss: {}, gate cross entropy loss: {}".format(classification_loss, gate_cel))
            flag = 1

        mem_loss = args.lambda_mem * gate_cel
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


def validate(testloader, model, criterion, criterion_gate, flag=0):
    """
    Run evaluation
    """
    model.eval()
    #end = time.time()
    test_loss = 0
    correct = 0
    total = 0
    test_loss_cel = 0.0
    test_loss_cls = 0.0
    with torch.no_grad():
        for i, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.to(device), target.to(device)
            output, g_validate = model(inputs)

            r = g_validate.size(0)
            # calculating the cross entropy loss for the gate
            '''
            Commented lines represent the code used for Binary cross entropy as loss for the gate.
            But binary cross entropy uses sigmoid and thats not appropriate for this case.
            We need softmax so that the sum of probabilities is 1
            
            #gate_target = torch.ones([r, 2], dtype=torch.float32)
            #gate_target[:,1] = 0.0
            '''
            gate_target = torch.zeros([r], dtype=torch.long)
            gate_target = gate_target.to(device)
            g_validate = g_validate.to(device)
            gate_cel = criterion_gate(g_validate, gate_target)

            classification_loss = criterion(output, target)
            mem_loss = args.lambda_mem * gate_cel
            loss = classification_loss + mem_loss

            test_loss += loss.item()
            test_loss_cls += classification_loss.item()
            test_loss_cel += gate_cel.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if not flag:
                progress_bar(i, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(i+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    return acc, test_loss/total, test_loss_cls/total, test_loss_cel/total

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the present LR decayed by lr_decay_gamma at every point in the schedule"""
    lr = 0.0
    if epoch == args.schedule[0]:
        lr = args.lr*(args.lr_decay_gamma**1)
    elif epoch == args.schedule[1]:
        lr = args.lr*(args.lr_decay_gamma**2)
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def save_checkpoint(state, name):
    """
    Save the training model
    """
    path = '/data/chercheurs/garg191/checkpoint/cifar_resnets_modified/' + p + '/c{}_{}_3248_{}_cel'.format(args.dataset[5:], args.arch+str(args.depth), x)
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))
    print("checkpoint saved: ", name)


if __name__ == '__main__':
    main()
