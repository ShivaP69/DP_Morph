
from backpack.core.derivatives.groupnorm import GroupNormDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase

class BatchGradGroupNorm(BatchGradBase):
    def __init__(self):
        super().__init__(GroupNormDerivatives())