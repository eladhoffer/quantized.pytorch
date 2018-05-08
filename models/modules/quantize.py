import torch
from torch.autograd.function import InplaceFunction
import torch.nn as nn
import torch.nn.functional as F
import math


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, inplace=False):
        if min_value is None:
            min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        if max_value is None:
            max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2.**num_bits - 1.

        scale = (max_value - min_value) / (qmax - qmin)
        if scale == 0:  # choose arbitrary scale
            scale = 1

        initial_zero_point = qmin - min_value / scale

        zero_point = 0.
        # make zero exactly represented
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)
        output.div_(scale).add_(zero_point).clamp_(
            qmin, qmax).round_()  # quantize
        output.add_(-zero_point).mul_(scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.view(
                grad_output.size(0), -1).min(-1)[0].mean())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.view(
                grad_output.size(0), -1).max(-1)[0].mean())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits,
                                             min_value, max_value, ctx.inplace)
        return grad_input, None, None, None, None


def quantize(x, num_bits=8, min_value=None, max_value=None, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, inplace)


def quantize_grad(x, num_bits=8, min_value=None, max_value=None, inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value))


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None

        output = F.conv2d(qinput, qweight, qbias, self.stride,
                          self.padding, self.dilation, self.groups)
        if self.num_bits_grad is not None:
            output = quantize_grad(output, num_bits=self.num_bits_grad)
        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                           min_value=float(self.weight.min()),
                           max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None

        output = F.linear(qinput, qweight, qbias)
        if self.num_bits_grad is not None:
            output = quantize_grad(output, num_bits=self.num_bits_grad)
        return output


class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_bits=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, x):
        # p=5
        K = 1
        NumOfChunks = 10
        C = 0
        x = self.quantize_input(x)
        if self.training:
            mean = x.view(x.size(0), x.size(self.dim), -1).mean(-1).mean(0)
            y = x.transpose(0, 1)
            z = y.contiguous()
            t = z.view(z.size(0), -1)
            A = t.transpose(1, 0) - mean

            B = torch.chunk(A, NumOfChunks, dim=0)

            for i in range(0, NumOfChunks):
                const = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(B[i].size(0))) ** 0.5)
                C = C + (torch.max(B[i], dim=0)[0] -
                         torch.min(B[i], dim=0)[0]) * const

            MeanTOPK = C / NumOfChunks

            scale = 1 / (MeanTOPK + 0.0000001)

            self.running_mean.mul_(self.momentum).add_(
                mean * (1 - self.momentum))

            self.running_var.mul_(self.momentum).add_(
                scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var

        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.weight is not None:
            qweight = quantize(self.weight, num_bits=self.num_bits,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()))
            out = out * qweight.view(1, qweight.size(0), 1, 1)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, qbias.size(0), 1, 1)
        return out
