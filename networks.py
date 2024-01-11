import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict
import math


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = self.s * F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self, in_dim=640, num_classes=10, scale=30, bias=True):
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out

class StridedConvNext(nn.Module):
    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConvNext, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, groups=32, bias=False),
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


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places,
                  kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class StridedConv(nn.Module):
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


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places *
                      self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places *
                          self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class NeXtBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, is_shortcut=False):
        super(NeXtBlock,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.is_shortcut = is_shortcut
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, groups=32,
                                   bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if is_shortcut:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_shortcut:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x


class ResNet_MoE(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet_MoE, self).__init__()
        self.expansion = expansion
        self.s = 30

        self.conv1 = Conv1(in_planes=3, places=64)

        # reduce dimension
        layer3_dim = 256 
        layer4_input_dim = 1024 
        layer4_dim = 512 
        self.layer1 = self.make_layer(
            in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(
            in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(
            in_places=512, places=layer3_dim, block=blocks[2], stride=2)
        self.layer4s = nn.ModuleList([self.make_layer(
            in_places=layer4_input_dim, places=layer4_dim, block=blocks[3], stride=2) for _ in range(3)])
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.norm_classifiers = nn.ModuleList([NormedLinear(2048, num_classes) for _ in range(3)])
        self.norm_rt_classifiers = nn.ModuleList([NormedLinear(2048, num_classes) for _ in range(3)])
        self.bias_classifiers = nn.ModuleList([Cos_Classifier(in_dim=2048, num_classes=num_classes) for _ in range(3)])
        self.bias_rt_classifiers = nn.ModuleList([Cos_Classifier(in_dim=2048, num_classes=num_classes) for _ in range(3)])
 
        feat_dim = 256
        planes = 128
        self.shallow_exp1 = nn.Sequential(
            OrderedDict([(f'SepConv{k}', StridedConv(in_planes=feat_dim * (2 ** k), planes=feat_dim * (2 ** (k + 1)), use_relu=(k != 2))) for
                         k in range(3)]))
        feat_dim *= 2
        planes *= 2
        self.shallow_exp2 = nn.Sequential(
            OrderedDict([(f'SepConv{k}', StridedConv(in_planes=feat_dim * (2 ** k), planes=feat_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(2)]))
        feat_dim *= 2
        planes *= 2
        self.shallow_exp3 = nn.Sequential(
            OrderedDict([(f'SepConv{k}', StridedConv(in_planes=feat_dim * (2 ** k), planes=feat_dim * (2 ** (k + 1)), use_relu=False)) for
                         k in range(1)]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x, crt=False):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        s1, s2, s3 = out1, out2, out3
        out3s = [out3, out3, out3]

        out4_1 = self.layer4s[0](out3s[0]) * self.shallow_exp1(s1)
        out4_2 = self.layer4s[1](out3s[1]) * self.shallow_exp2(s2)
        out4_3 = self.layer4s[2](out3s[2]) * self.shallow_exp3(s3)

        out4_1 = self.avgpool(out4_1)
        out4_2 = self.avgpool(out4_2)
        out4_3 = self.avgpool(out4_3)

        flattened_feature1 = out4_1.view(out4_1.size(0), -1)
        flattened_feature2 = out4_2.view(out4_2.size(0), -1)
        flattened_feature3 = out4_3.view(out4_3.size(0), -1)
        
        feats = [flattened_feature1, flattened_feature2, flattened_feature3]
        outs = []
        if crt == True:
            for i in range(3):
                outs.append(self.norm_rt_classifiers[i](feats[i]))
        else:
            for i in range(3):
                outs.append(self.norm_classifiers[i](feats[i]))
        return outs


class ResNeXt_MoE(nn.Module):
    def __init__(self, blocks, num_classes=1000):
        super(ResNeXt_MoE, self).__init__()
        self.s = 30

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(
            in_planes=64, planes=256, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(
            in_planes=256, planes=512, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(
            in_planes=512, planes=1024, block=blocks[2], stride=2)
        self.layer4s = nn.ModuleList([self.make_layer(
            in_planes=1024, planes=2048, block=blocks[3], stride=2) for _ in range(3)])
        

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.norm_classifiers = nn.ModuleList([NormedLinear(2048, num_classes) for _ in range(3)])
        self.norm_rt_classifiers = nn.ModuleList([NormedLinear(2048, num_classes) for _ in range(3)])
        self.bias_classifiers = nn.ModuleList([Cos_Classifier(in_dim=2048, num_classes=num_classes) for _ in range(3)])
        self.bias_rt_classifiers = nn.ModuleList([Cos_Classifier(in_dim=2048, num_classes=num_classes) for _ in range(3)])
 
        feat_dim = 256
        self.shallow_exp1 = nn.Sequential(
            OrderedDict([(f'SepConv{k}', StridedConvNext(in_planes=feat_dim * (2 ** k), planes=feat_dim * (2 ** (k + 1)), use_relu=(k != 2))) for
                         k in range(3)]))
        feat_dim *= 2
        self.shallow_exp2 = nn.Sequential(
            OrderedDict([(f'SepConv{k}', StridedConvNext(in_planes=feat_dim * (2 ** k), planes=feat_dim * (2 ** (k + 1)), use_relu=(k != 1))) for
                         k in range(2)]))
        feat_dim *= 2
        self.shallow_exp3 = nn.Sequential(
            OrderedDict([(f'SepConv{k}', StridedConvNext(in_planes=feat_dim * (2 ** k), planes=feat_dim * (2 ** (k + 1)), use_relu=False)) for
                         k in range(1)]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_planes, planes, block, stride):
        layers = []
        layers.append(NeXtBlock(in_planes, planes, stride, is_shortcut=True))
        for i in range(1, block):
            layers.append(NeXtBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, crt=False):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        s1, s2, s3 = out1, out2, out3
        out3s = [out3, out3, out3]

        out4_1 = self.layer4s[0](out3s[0]) * self.shallow_exp1(s1)
        out4_2 = self.layer4s[1](out3s[1]) * self.shallow_exp2(s2)
        out4_3 = self.layer4s[2](out3s[2]) * self.shallow_exp3(s3)

        out4_1 = self.avgpool(out4_1)
        out4_2 = self.avgpool(out4_2)
        out4_3 = self.avgpool(out4_3)

        flattened_feature1 = out4_1.view(out4_1.size(0), -1)
        flattened_feature2 = out4_2.view(out4_2.size(0), -1)
        flattened_feature3 = out4_3.view(out4_3.size(0), -1)
        
        feats = [flattened_feature1, flattened_feature2, flattened_feature3]
        outs = []
        if crt == True:
            for i in range(3):
                outs.append(self.norm_rt_classifiers[i](feats[i]))
        else:
            for i in range(3):
                outs.append(self.norm_classifiers[i](feats[i]))
        return outs


def ResNet50_MoE():
    print("=> creating model ResNet50_MoE")
    return ResNet_MoE([3, 4, 6, 3])

def ResNeXt50_MoE():
    print("=> creating model ResNeXt50_MoE")
    return ResNeXt_MoE([3, 4, 6, 3])


def ResNet101():
    return ResNet_MoE([3, 4, 23, 3])


def ResNet152():
    return ResNet_MoE([3, 8, 36, 3])


if __name__=='__main__':
    model = ResNet50_MoE()
    print(model)
    for param in model.named_parameters():
        print(param[0])
