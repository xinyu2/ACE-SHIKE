'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
This MoE design is based on the implementation of Yerlan Idelbayev.
'''

from collections import OrderedDict
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


def _weights_init(m):
    classname = m.__class__.__name__ #This line retrieves the name of the class to which the instance m belongs. For example, if m is an instance of the nn.Linear class, classname will be 'Linear'.
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #This line checks whether the object m is an instance of either nn.Linear or nn.Conv2d.
        init.kaiming_normal_(m.weight)
  
"""
The LambdaLayer class is a flexible and general-purpose layer that can perform any operation defined by the lambd function. 
This allows for easy customization of neural network layers without having to define a new class for each specific operation. 
The layer can be used in a neural network architecture by creating an instance of LambdaLayer and providing a function that defines the desired operation.
"""
class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        #The constructor takes one parameter, lambd, which is expected to be a function.
        self.lambd = lambd
        """
        The function self.lambd is expected to be a callable that takes a tensor x as input and returns a tensor. This makes LambdaLayer very flexible, as it can perform any operation on x that is defined in the lambd function.
        """
    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    #control the expansion of channels in the block. the output channel equals to the input  channnel, where expansion =1.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:#throwaway functions that are not needed elsewhere in the code,
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",#::2, select all the elements along the height
                                                  0))#//returns the largest integer less than or equal to the result.
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):

    """
    resduial block for resnet 50

    """
    expansion = 4

    def __init__(self, in_planes, planes, stride =1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding =1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size = 1, stride = 1, padding =1, bias =False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)# more efficient memory useage of input.
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes,
                      kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        identity  = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)

        return out
         



class StridedConv(nn.Module):
    """
    downsampling conv layer
    """

    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)

        if self.use_relu:
            out = self.relu(out)

        return out


class ShallowExpert(nn.Module):
    """
    shallow features alignment wrt. depth
    """

    def __init__(self, input_dim=None, depth=None) -> None:
        super(ShallowExpert, self).__init__()
        self.convs = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(in_planes=input_dim * (2 ** k), planes=input_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(depth)]))

    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class ResNet_MoE50(nn.Module):
    def __init__(self,block, num_blocks, num_experts=None, num_classes =1000, zero_init_residual=False, groups=1, 
                  width_per_group=64, norm_layer=None, replace_stride_with_dilation=None, use_norm = False):
        super(ResNet_MoE50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.s = 1
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.base_width = width_per_group
        self.num_experts = num_experts
        self.in_planes = 64
        self.next_in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, num_blocks[2],stride=2,
                                       dilate=replace_stride_with_dilation[1])

        if num_experts:
            layer4_output_dim = 512
            self.in_planes = 256
            self.layer4s = nn.ModuleList([self._make_layer(
                block, layer4_output_dim, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2]) for _ in range(self.num_experts)])
            self.in_planes = self.next_in_planes
            if use_norm:
                self.s =30
                self.classifiers = nn.ModuleList(
                    [NormedLinear(512, num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(512, num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers =nn.ModuleList(
                    [NormedLinear(512, num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dilate=replace_stride_with_dilation[2])
            self.linear = NormedLinear(512, num_classes) if use_norm else nn.Linear(
                512, num_classes, bias=True)
        self. apply(_weights_init)
        self.depth = list(
            reversed([i + 1 for i in range(len(num_blocks) - 1)]))
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(
            self.num_experts)]
        feat_dim = 64
        self.shallow_exps = nn.ModuleList([ShallowExpert(
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])
        self.shallow_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer =self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.next_in_planes = self.in_planes
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.next_in_planes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.next_in_planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)



        """ 
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)
        """

    def forward(self, x, crt=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        shallow_outs = [out1, out2, out3]

        if self.num_experts:
            out4s = [self.layer4s[_](out3) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]

            exp_outs = [out4s[i] * shallow_expe_outs[i]
                        for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
        else:
            out4 = self.layer4(out3)
            out = F.avg_pool2d(out4, out4.size()[3]).view(out4.size(0), -1)
            outs = self.linear(out)

        return outs


class ResNet_MoE(nn.Module):

    def __init__(self, block, num_blocks, num_experts=None, num_classes=10, use_norm=False):
        super(ResNet_MoE, self).__init__()
        self.s = 1
        self.num_experts = num_experts
        self.in_planes = 16
        self.next_in_planes = 16 # typically update in _make_layer method.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        if num_experts:
            layer3_output_dim = 64
            self.in_planes = 32
            self.layer3s = nn.ModuleList([self._make_layer(
                block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(self.num_experts)])
            self.in_planes = self.next_in_planes
            if use_norm:
                self.s = 30
                self.classifiers = nn.ModuleList(
                    [NormedLinear(64, num_classes) for _ in range(self.num_experts)])
                self.rt_classifiers = nn.ModuleList(
                    [NormedLinear(64, num_classes) for _ in range(self.num_experts)])
            else:
                self.classifiers = nn.ModuleList(
                    [nn.Linear(64, num_classes, bias=True) for _ in range(self.num_experts)])
        else:
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = NormedLinear(64, num_classes) if use_norm else nn.Linear(
                64, num_classes, bias=True)

        self.apply(_weights_init)
        self.depth = list(
            reversed([i + 1 for i in range(len(num_blocks) - 1)]))  # [2, 1]
        self.exp_depth = [self.depth[i % len(self.depth)] for i in range(
            self.num_experts)]  # [2, 1, 2]
        feat_dim = 16
        self.shallow_exps = nn.ModuleList([ShallowExpert(
            input_dim=feat_dim * (2 ** (d % len(self.depth))), depth=d) for d in self.exp_depth])

        self.shallow_avgpool = nn.AdaptiveAvgPool2d((8, 8))

    def _make_layer(self, block, planes, num_blocks, stride):# resnet comprises severl layers, each consisting severl blocks.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, crt=False):

        out = F.relu(self.bn1(self.conv1(x)))

        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        shallow_outs = [out1, out2]
        if self.num_experts:
            out3s = [self.layer3s[_](out2) for _ in range(self.num_experts)]
            shallow_expe_outs = [self.shallow_exps[i](
                shallow_outs[i % len(shallow_outs)]) for i in range(self.num_experts)]

            exp_outs = [out3s[i] * shallow_expe_outs[i]
                        for i in range(self.num_experts)]
            exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                output.size(0), -1) for output in exp_outs]
            if crt == True:
                outs = [self.s * self.rt_classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
            else:
                outs = [self.s * self.classifiers[i]
                        (exp_outs[i]) for i in range(self.num_experts)]
        else:
            out3 = self.layer3(out2)
            out = F.avg_pool2d(out3, out3.size()[3]).view(out3.size(0), -1)
            outs = self.linear(out)

        return outs

# for Cifar100-LT use


def resnet32(num_classes=100, use_norm=False, num_exps=None):
    return ResNet_MoE(BasicBlock, [5, 5, 5], num_experts=num_exps, num_classes=num_classes, use_norm=use_norm)

    
def resnet50(num_classes=1000, use_norm=False, num_exps=None):
    return ResNet_MoE50(Bottleneck, [3, 4, 6, 3], num_experts=num_exps, num_classes=num_classes, use_norm=use_norm)


def test(net):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params:", total_params)
    print("Total layers:", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    moe32 = resnet32(num_classes=100, num_exps=3, use_norm=False)
    test(net=moe32)
