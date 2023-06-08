import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.backends
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from math import pi
import math
import numpy as np
import matplotlib.pyplot as plt

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        f = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        '''
        f中的每个元素的size分别是 bs 256 w/4 h/4， bs 512 w/8 h/8， 
        bs 1024 w/16 h/16， bs 2048 w/32 h/32
        '''
        return f

def resnet9(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        
        #model.load_state_dict(torch.load("./resnet50-19c8e357.pth"))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        
        #model.load_state_dict(torch.load("./resnet50-19c8e357.pth"))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        
        #model.load_state_dict(torch.load("./resnet50-19c8e357.pth"))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        
        #model.load_state_dict(torch.load("./resnet50-19c8e357.pth"))
#         model.load_state_dict(torch.load("./unet/resnet101.pth"))
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained=False, progress=True, **kwargs):
    
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class dfpn(nn.Module):
    def __init__(self, backbone_pretrained = True, trans_channel_num = 128, resnet_type = 'resnet18', resnet_layer = [64,128,256,512]):
        super(dfpn,self).__init__()
        if resnet_type == 'resnet9':
            self.resnet = resnet9(backbone_pretrained)
        if resnet_type == 'resnet18':
            self.resnet = resnet18(backbone_pretrained)
        if resnet_type == 'resnet34':
            self.resnet = resnet34(backbone_pretrained)
        if resnet_type == 'resnet50':
            self.resnet = resnet50(backbone_pretrained)
        if resnet_type == 'resnet101':
            self.resnet = resnet101(backbone_pretrained)
        if resnet_type == 'resnet152':
            self.resnet = resnet152(backbone_pretrained)
            
        conv1_inchannel_num = trans_channel_num*2
        conv2_inchannel_num = trans_channel_num*3
        conv3_inchannel_num = trans_channel_num*4
 
        self.conv1 = nn.Conv2d(conv1_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn1 = nn.BatchNorm2d(trans_channel_num)
        
        self.conv2 = nn.Conv2d(conv2_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn2 = nn.BatchNorm2d(trans_channel_num)
        
        self.conv3 = nn.Conv2d(conv3_inchannel_num, trans_channel_num, 3,padding=1)
        self.bn3 = nn.BatchNorm2d(trans_channel_num)
        
        self.unpool = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.convl1 = nn.Conv2d(resnet_layer[0],trans_channel_num,1)
        self.convl2 = nn.Conv2d(resnet_layer[1],trans_channel_num,1)
        self.convl3 = nn.Conv2d(resnet_layer[2],trans_channel_num,1)
        self.convl4 = nn.Conv2d(resnet_layer[3],trans_channel_num,1)
    def forward(self,images):
        f = self.resnet(images)
        c1 = self.convl1(f[0])
        c2 = self.convl2(f[1])
        c3 = self.convl3(f[2])
        c4 = self.convl4(f[3])

        p4 = self.unpool(c4)
        p3 = self.conv1(torch.cat((p4,c3), 1))
        # p3 = self.bn1(p3)
        
        p3 = self.unpool(p3)
        p4 = self.unpool(p4)
        p2 = self.conv2(torch.cat((p3,p4,c2), 1))
        # p2 = self.bn2(p2)

        p2 = self.unpool(p2)
        p3 = self.unpool(p3)
        p4 = self.unpool(p4)
        p1 = self.conv3(torch.cat((p2,p3,p4,c1), 1))
        # p1 = self.bn3(p1)
        
        return p1