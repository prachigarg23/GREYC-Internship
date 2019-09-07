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


__all__ = ['resnet_exg_1by1element']

def conv3x3(in_planes, out_planes, stride=1, kernel_size=3, bias=False, pad=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                     bias=bias, padding=pad)


class BasicBlock(nn.Module):
    expansion = 1
#    loss_g = torch.tensor([0.0], device='cuda', requires_grad=True) #static variable to store the g norm
    loss_g = 0.0
    criterion_gate = nn.CrossEntropyLoss()

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.gate_layer = conv3x3(inplanes, planes*BasicBlock.expansion*2, stride=self.stride, kernel_size=1, bias=True, pad=0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        g = self.gate_layer(x)
        bs, c, n, _ = g.shape
        f = c//2
        g1 = g[:,:f,:,:]
        g2 = g[:,f:,:,:]
        g1 = g1.contiguous().view(-1, 1)
        g2 = g2.contiguous().view(-1, 1)
        g = torch.cat((g1, g2), 1)

        gate_target = torch.ones([bs*f*n*n], dtype=torch.long)
        #gate_target = torch.ones([bs*f*n*n, 2], dtype=torch.float32)
        #gate_target[:,0] = 0.0
        gate_target = gate_target.to('cuda')
        g = g.to('cuda')
        gate_cel = BasicBlock.criterion_gate(g, gate_target)
        BasicBlock.loss_g += gate_cel

        g_gate = F.softmax(g*0.0001, dim=1)
        #print('check 5 ', torch.sum(g_gate,0))
        #print('check 5 ', g_gate)
        g1 = g_gate[:,0]
        g2 = g_gate[:,1]
        g1 = g1.view(bs, f, n, n)
        g2 = g2.view(bs, f, n, n)

        output = g1*out + g2*residual #element wise matrix multiplication as g1, g2, out and residual all are of same dimensions

        output = self.relu(output)
        return output


class Bottleneck(nn.Module):
    expansion = 4
    loss_g = 0.0
    criterion_gate = nn.CrossEntropyLoss()

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
        self.gate_layer = conv3x3(inplanes, planes*Bottleneck.expansion*2, stride=self.stride, kernel_size=1, bias=True, pad=0)


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

        g = self.gate_layer(x)
        bs, c, n, _ = g.shape
        f = c//2
        g1 = g[:,:f,:,:]
        g2 = g[:,f:,:,:]
        g1 = g1.contiguous().view(-1, 1)
        g2 = g2.contiguous().view(-1, 1)
        g = torch.cat((g1, g2), 1)

        gate_target = torch.ones([bs*f*n*n], dtype=torch.long)
        gate_target = gate_target.to('cuda')
        g = g.to('cuda')
        gate_cel = Bottleneck.criterion_gate(g, gate_target)
        Bottleneck.loss_g += gate_cel

        g_gate = F.softmax(g*0.0001, dim=1)
        g1 = g_gate[:,0]
        g2 = g_gate[:,1]
        g1 = g1.view(bs, f, n, n)
        g2 = g2.view(bs, f, n, n)

        output = g1*out + g2*residual #element wise matrix multiplication as g1, g2, out and residual all are of same dimensions


        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=10, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if not isinstance(m.bias, type(None)):
                    BIAS_INIT = -6
                    m.bias.data.fill_(BIAS_INIT)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_exg_1by1element(**kwargs):
    """
    Constructs a ResNet model.
    """
    print('\nHi, inside resnet_exg_1by1element\n')
    return ResNet(**kwargs)
