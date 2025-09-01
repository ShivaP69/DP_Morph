# extensions/custom_group_norm.py

import torch
from torch.nn import GroupNorm
from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils.subsampling import subsample

class GroupNormDerivatives(BaseDerivatives):
    def __init__(self):
        super().__init__()

    def get_module(self):
        return GroupNorm

    def hessian_is_psd(self):
        return False

    def df_dx(self, module, g_inp, g_out, backproped):
        return backproped

    def df_dw(self, module, g_inp, g_out, backproped):
        return self.df_dx(module, g_inp, g_out, backproped)

    def df_db(self, module, g_inp, g_out, backproped):
        return self.df_dx(module, g_inp, g_out, backproped)


class BatchGradGroupNorm(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        return self._jacobian_weight(module, g_inp, g_out)

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        return self._jacobian_bias(module, g_inp, g_out)

    def _jacobian_weight(self, module, g_inp, g_out):
        N, C = module.input0.shape[0], module.num_channels
        grad_weight = (
            g_out[0]
            .transpose(0, 1)
            .reshape(C, N, -1)
            .sum(dim=2)
            .reshape(C, N)
        )
        return grad_weight

    def _jacobian_bias(self, module, g_inp, g_out):
        N, C = module.input0.shape[0], module.num_channels
        grad_bias = g_out[0].transpose(0, 1).reshape(C, N, -1).sum(dim=2).reshape(C, N)
        return grad_bias

# Register the custom extension
extensions = {"GroupNorm": BatchGradGroupNorm}
