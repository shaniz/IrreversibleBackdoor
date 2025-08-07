import copy
import timm
import torch.backends.cudnn as cudnn
import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth'))
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    return model


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #torch.nn.ModuleDict(features)
        self.layer_n = len(features)
        self.classifier1 = nn.Sequential(
        #self.classifier = nn.Sequential(
            nn.Linear(2048, 256),#*7*7  #cifar(32*32)则是512 cifar(64*64)2048#imagenet(224*224) 25088
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        
        #self.c1 = nn.Conv2d(64, 64, kernel_size=1, padding=1, groups=64)

    def forward(self, x, y=None):
        if y == None:
            x = self.features[:10](x)
            x = self.features[10:](x)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)
            return x
        else:
            x0 = self.features[:10](x)
            x = self.features[10:](x0)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)

            y0 = self.features[:10](y)
            y = self.features[10:](y0)
            y = y.view(y.size(0), -1)
            y = self.classifier1(y)
            return x, y, x0, y0

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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i in range(len(cfg)):
        v = cfg[i]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict = False)
    return model


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def process(checkpoint):
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def get_pretrained_model(arch, model_path, partial_finetuned=False):
    if arch == 'caformer':
        model = timm.create_model("caformer_m36", pretrained=False)
        classifier = nn.Linear(2304, 10)
        model.head.fc.fc2 = classifier
        model.load_state_dict(process(torch.load(model_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.fc2.parameters():
                param.requires_grad = True
        return model.cuda()

    elif arch == 'vgg':
        model = VGG(make_layers(cfg['B']), num_classes=10)
        model.load_state_dict(process(torch.load(model_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif arch == 'res18':
        model = resnet18(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(model_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif arch == 'res34':
        model = resnet34(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(model_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif arch == 'res50':
        model = resnet50(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(model_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    else:
        assert (0)


def save_bn(model):
    means = []
    vars = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            means.append(copy.deepcopy(layer.running_mean))
            vars.append(copy.deepcopy(layer.running_var))

    return means, vars


def load_bn(model, means, vars):
    idx = 0
    for _, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean = copy.deepcopy(means[idx])
            layer.running_var = copy.deepcopy(vars[idx])
            idx += 1
    return model