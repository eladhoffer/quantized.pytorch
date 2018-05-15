
import torch
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable
import torch.nn as nn
import math


class BiReLUFunction(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, inplace=False):
        if input.size(1) % 2 != 0:
            raise RuntimeError("dimension 1 of input must be multiple of 2, "
                               "but got {}".format(input.size(1)))
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        pos, neg = output.chunk(2, dim=1)
        pos.clamp_(min=0)
        neg.clamp_(max=0)
        # scale = (pos - neg).view(pos.size(0), -1).mean(1).div_(2)
        # output.
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables
        grad_input = grad_output.masked_fill(output.eq(0), 0)
        return grad_input, None


def birelu(x, inplace=False):
    return BiReLUFunction().apply(x, inplace)


class BiReLU(nn.Module):
    """docstring for BiReLU."""

    def __init__(self, inplace=False):
        super(BiReLU, self).__init__()
        self.inplace = inplace

    def forward(self, inputs):
        return birelu(inputs, inplace=self.inplace)


def binorm(x, shift=0, scale_fix=(2 / math.pi) ** 0.5):
    pos, neg = (x + shift).split(2, dim=1)
    scale = (pos - neg).view(pos.size(0), -1).mean(1).div_(2) * scale_fix
    return x / scale


def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


def rnlu(x, inplace=False, shift=0, scale_fix=(math.pi / 2) ** 0.5):
    x = birelu(x, inplace=inplace)
    pos, neg = (x + shift).chunk(2, dim=1)
    # scale = torch.cat((_mean(pos, 1), -_mean(neg, 1)), 1) * scale_fix + 1e-5
    scale = (pos - neg).view(pos.size(0), -1).mean(1) * scale_fix + 1e-8
    return x / scale.view(scale.size(0), *([1] * (x.dim() - 1)))


class RnLU(nn.Module):
    """docstring for RnLU."""

    def __init__(self, inplace=False):
        super(RnLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return rnlu(x, inplace=self.inplace)

    # output.
if __name__ == "__main__":
    x = Variable(torch.randn(2, 16, 5, 5).cuda(), requires_grad=True)
    output = rnlu(x)

    output.sum().backward()
