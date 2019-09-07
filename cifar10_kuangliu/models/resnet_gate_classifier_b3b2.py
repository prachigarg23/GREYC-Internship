from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

__all__ = ['resnet_gate_classifier']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 164, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.num_classes = num_classes
        print('\n\ncheck, num_classes ', self.num_classes)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)

        self.o21, self.layer2, self.o22 = self._make_layer2(block, 32, n, stride=2)
        self.g_layer_b2 = nn.Sequential(nn.Linear(32*block.expansion, 216), nn.Linear(216, num_classes),)
        self.avgpool_b2 = nn.AvgPool2d(16)

        self.o31, self.layer3, self.o32 = self._make_layer2(block, 64, n, stride=2)
        self.avgpool_b3 = nn.AvgPool2d(8)
        self.g_layer_b3 = nn.Sequential(nn.Linear(64*block.expansion, 512), nn.Linear(512, num_classes),)

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        o1 = layers[0]
        o2 = layers[-1]
        return o1, nn.Sequential(*layers[1:-1]), o2

    def calculate_g(self, g_layer_out):
        # g_layer_out being used to calculate the entropy which
        # will be used as g in the gate at both test and train time
        g = F.softmax(g_layer_out, dim=1) * F.log_softmax(g_layer_out, dim=1)
        g = -1.0 * g.sum(1)

        if self.num_classes == 10:
            bias = 2.3026/2
        elif self.num_classes == 100:
            bias = 4.6052/2
        else:
            print("something's seriously wrong check the code.............")
        g = g - bias
        g = torch.sigmoid(g)

        return g

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        #B1
        x = self.layer1(x)  # 32x32

        #B2
        o21 = self.o21(x)
        x = self.layer2(o21)  # 16x16
        o22 = self.o22(x)

        ox_b2 = self.avgpool_b2(o21)
        o21_ = ox_b2.view(ox_b2.size(0), -1)
        g_layer_out_b2 = self.g_layer_b2(o21_)
        
        g2 = self.calculate_g(g_layer_out_b2)
        x = g2[:, None, None, None]*o22+(1-g2[:,None,None,None])*o21

        #B3
        o31 = self.o31(x)
        x = self.layer3(o31)  # 8x8
        o32 = self.o32(x)

        ox_b3 = self.avgpool_b3(o31)
        o31_ = ox_b3.view(ox_b3.size(0), -1)
        g_layer_out_b3 = self.g_layer_b3(o31_)

        g3 = self.calculate_g(g_layer_out_b3)
        x = g3[:, None, None, None]*o32+(1-g3[:,None,None,None])*o31

        x = self.avgpool_b3(x) # this needs avgpool(8) hence used avgpool_b3
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        '''
        the out will be used to get classification cross entropy loss
        the g_layer_out_b2 and g_layer_out_b3 will be used to calc gate classifier cross entropy loss which will only update the gate layers
        '''
        return x, g_layer_out_b2, g_layer_out_b3

def resnet_gate_classifier(**kwargs):
    """
    Constructs a ResNet model.
    """
    print("\nHi, I'm ready to use the gate over 2 blocks, B2 and the last block B3!\n")
    return ResNet(**kwargs)
