import torch


# TODO
class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight)


class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return torch.nn.functional.linear(quantized_act, quantized_weight, self.bias)


QuanModuleMapping = {
    torch.nn.Conv2d: QuanConv2d,
    torch.nn.Linear: QuanLinear
}