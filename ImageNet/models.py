import torch
import torch.nn as nn
import sys
sys.path.append("..")
from ..util import BinLinear
from ..util import BinConv2d


class XNOR_VGG(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(XNOR_VGG_D, self).__init__()
        self.name = 'XNOR_VGG'
        cfg = [[3, 64], [64, 64], ['M'],
               [64, 128], [128, 128], ['M'],
               [128, 256], [256, 256], [256, 256], ['M'],
               [256, 512], [512, 512], [512, 512], ['M'],
               [512, 512], [512, 512], [512, 512], ['M']]
        self.features = make_bin_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            BinLinear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            BinLinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),)
        if init_weights:
            self._initialize_weights()

    def forward(self, *input):
        x = input[0]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_bin_layers(cfg):
    layers = []
    for v in cfg:
        if v == ['M']:
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == [3, 64]:
            conv2d = nn.Conv2d(v[0], v[1], kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
        else:
            conv2d = BinConv2d(v[0], v[1], kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
