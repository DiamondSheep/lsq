import torch
import torch.nn as nn
import copy
from .quant_module import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock


def quantize_model(model):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """

    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=8)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=8)
        quant_mod.set_param(model)
        return quant_mod

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
        return nn.Sequential(*[model, QuantAct(activation_bit=8)])

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m))
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, quantize_model(mod))
        return q_model


def freeze_model(model):
    """
    freeze the activation range
    """
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model


def unfreeze_model(model):
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return 