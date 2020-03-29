#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from ..ops import emulate_int


class IntConv2d(_ConvNd):
    """
    Quantized counterpart of the nn.Conv2d module that applies QuantNoise during training.

    Args:
        - standard nn.Conv2d parameters
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iteartions

    Remarks:
        - We use the straight-thgourh estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        p=0,
        bits=8,
        method="histogram",
        update_step=1000,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(IntConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )

        # quantization parameters
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0
        self.scale_activations = None
        self.zero_point_activations = None

    def _conv_forward(self, input, weight):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        # train with QuantNoise and evaluate the fully quantized network
        p = self.p if self.training else 1

        # update parameters every 100 iterations
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1

        # quantize weight
        weight_quantized, self.scale, self.zero_point = emulate_int(
            self.weight.detach(),
            bits=self.bits,
            method=self.method,
            scale=self.scale,
            zero_point=self.zero_point,
        )

        # mask to apply noise
        mask = torch.zeros_like(self.weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)

        # using straight-through estimator (STE)
        weight = self.weight + noise.detach()

        # return output
        output = self._conv_forward(input, weight)
        return output

    def extra_repr(self):
        return (
            "in_channels={}, out_channels={}, kernel_size={}, stride={}, "
            "padding={}, dilation={}, groups={}, bias={}, dropout={}, "
            "bits={}, method={}".format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.bias is not None,
                self.p,
                self.bits,
                self.method,
            )
        )