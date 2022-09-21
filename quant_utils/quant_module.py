import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Parameter
from .quant_utils import *
from .quant_base_module import *

class LSQ(torch.autograd.Function):
    # https://github.com/hustzxd/LSQuantization/blob/master/STE_LSQ.ipynb
    """
    Usage:
        w_q = LSQ.apply(self.weight, self.alpha, g, Qn, Qp)
    """
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        """ Args:
         - alpha: quantization scale
         - g: gradient step to correct the step size
         - Qn: low bound
         - Qp: high bound
        """
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        quant_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = quant_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None

class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_bit=4, mode=Qmodes.layer_wise):
        super(Conv2dLSQ, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            weight_bit=weight_bit, mode=mode)
    
    def set_param(self, conv: nn.Conv2d):
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    @staticmethod
    def get_param(module: nn.Conv2d):
        params = [module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, True]
        return params

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.weight_bit - 1)
        Qp = 2 ** (self.weight_bit - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        w_q = LSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, weight_bit=4):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, weight_bit=weight_bit)
    
    def set_param(self, linear: nn.Linear):
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    @staticmethod
    def get_param(module: nn.Linear):
        params = [module.in_features, module.out_features, True]
        return params

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.weight_bit - 1)
        Qp = 2 ** (self.weight_bit - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        w_q = LSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)

class ActLSQ(_ActQ):
    def __init(self, act_bit=4, signed=False):
        super(ActLSQ, self).__init(act_bit=act_bit, signed=signed)

    def forward(self, x):
        if self.alpha is None:
            return x
        if self.signed:
            Qn = -2 ** (self.act_bit - 1)
            Qp = 2 ** (self.act_bit - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.act_bit - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        x_q = LSQ.apply(x, self.alpha, g, Qn, Qp)
        return x

class QuantAct(Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 running_stat=True):
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.act_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def forward(self, x):
        """
        quantize given activation x
        """
        if self.running_stat:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max)

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min,
                                          self.x_max)
            return quant_act
        else:
            return x


class Quant_Linear(Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.detach()
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d(Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = AsymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        w = self.weight
        x_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = x_transform.min(dim=1).values
        w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min,
                                     w_max)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)