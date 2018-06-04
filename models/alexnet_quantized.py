import torch.nn as nn
import torchvision.transforms as transforms
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN

__all__ = ['alexnet_quantized']

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 8
BIPRECISION = True


class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(
            QConv2d(3, 64, kernel_size=11, stride=4, padding=2,
                    bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(kernel_size=3, stride=2),
            RangeBN(64),
            nn.ReLU(inplace=True),
            QConv2d(64, 192, kernel_size=5, padding=2, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            RangeBN(192),
            QConv2d(192, 384, kernel_size=3, padding=1, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.ReLU(inplace=True),
            RangeBN(384),
            QConv2d(384, 256, kernel_size=3, padding=1, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.ReLU(inplace=True),
            RangeBN(256),
            QConv2d(256, 256, kernel_size=3, padding=1, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            RangeBN(256)
        )
        self.classifier = nn.Sequential(
            QLinear(256 * 6 * 6, 4096, bias=False, num_bits=NUM_BITS,
                    num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            QLinear(4096, 4096, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT,
                    num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION),
            RangeBN(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            QLinear(4096, num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT,
                    num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        )

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
             'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-3},
            {'epoch': 60, 'lr': 1e-4},
            {'epoch': 90, 'lr': 5e-4}]
        ]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_quantized(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return AlexNetOWT_BN(num_classes)
