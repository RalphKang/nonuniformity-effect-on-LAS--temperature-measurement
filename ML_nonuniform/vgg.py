"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from torchsummary import summary

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=11):
        super().__init__()
        self.features = features
        self.final_feature=nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output =self.final_feature(output)
        # print(output.size())
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 1 # change input channel to 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv1d(input_channel, l, kernel_size=3, padding=0)]

        if batch_norm:
            layers += [nn.BatchNorm1d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_NObn():
    return VGG(make_layers(cfg['A'], batch_norm=False))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))



# net=VGG(make_layers(cfg['A'], batch_norm=False),5)
# # net.to("cuda")
# summary(net,batch_size=1, input_size=(1,201),device='cpu')