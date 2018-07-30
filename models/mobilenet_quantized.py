import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
__all__ = ['mobilenet_quantized']

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 8
BIPRECISION = True


def nearby_int(n):
    return int(round(n))


def init_model(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, RangeBN):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()


class DepthwiseSeparableFusedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super(DepthwiseSeparableFusedConv2d, self).__init__()
        self.components = nn.Sequential(
            QConv2d(in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, groups=in_channels, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(in_channels, num_bits=NUM_BITS,
                    num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(),

            QConv2d(in_channels, out_channels, 1, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(out_channels, num_bits=NUM_BITS,
                    num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU()
        )

    def forward(self, x):
        return self.components(x)


class MobileNet(nn.Module):

    def __init__(self, width=1., shallow=False, num_classes=1000):
        super(MobileNet, self).__init__()
        num_classes = num_classes or 1000
        width = width or 1.
        layers = [
            QConv2d(3, nearby_int(width * 32),
                    kernel_size=3, stride=2, padding=1, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(nearby_int(width * 32), num_bits=NUM_BITS,
                    num_bits_grad=NUM_BITS_GRAD),
            nn.ReLU(inplace=True),

            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 32), nearby_int(width * 64),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 64), nearby_int(width * 128),
                kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 128), nearby_int(width * 128),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 128), nearby_int(width * 256),
                kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 256), nearby_int(width * 256),
                kernel_size=3, padding=1),
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 256), nearby_int(width * 512),
                kernel_size=3, stride=2, padding=1)
        ]
        if not shallow:
            # 5x 512->512 DW-separable convolutions
            layers += [
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
                DepthwiseSeparableFusedConv2d(
                    nearby_int(width * 512), nearby_int(width * 512),
                    kernel_size=3, padding=1),
            ]
        layers += [
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 512), nearby_int(width * 1024),
                kernel_size=3, stride=2, padding=1),
            # Paper specifies stride-2, but unchanged size.
            # Assume its a typo and use stride-1 convolution
            DepthwiseSeparableFusedConv2d(
                nearby_int(width * 1024), nearby_int(width * 1024),
                kernel_size=3, stride=1, padding=1)
        ]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = QLinear(nearby_int(width * 1024), num_classes, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3},
            {'epoch': 80, 'lr': 1e-4}
        ]

    @staticmethod
    def regularization(model, weight_decay=4e-5):
        l2_params = 0
        for m in model.modules():
            if isinstance(m, QConv2d) or isinstance(m, nn.Linear):
                l2_params += m.weight.pow(2).sum()
                if m.bias is not None:
                    l2_params += m.bias.pow(2).sum()
        return weight_decay * 0.5 * l2_params

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet_quantized(**kwargs):
    r"""MobileNet model architecture from the `"MobileNets:
    Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_classes, width, alpha, shallow = map(
        kwargs.get, ['num_classes', 'width', 'alpha', 'shallow'])
    return MobileNet(width=width, shallow=shallow, num_classes=num_classes)
