import torch
from lightning.pytorch import Callback


class SetPrecisionAndCudnnCallback(Callback):
    """
    Set fp32 and fp16 precision and cudnn when training starts. For details, see
    https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    https://pytorch.org/docs/stable/generated/torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction.html
    https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html#torch.set_deterministic_debug_mode
    https://pytorch.org/docs/stable/backends.html#module-torch.backends.cudnn
    """

    def __init__(
        self,
        float32_matmul_precision=None,
        allow_fp16_reduced_precision_reduction=None,
        deterministic_debug_mode=None,
        cudnn_enabled=None,
    ):
        self.float32_matmul_precision = float32_matmul_precision
        self.allow_fp16_reduced_precision_reduction = (
            allow_fp16_reduced_precision_reduction
        )
        self.deterministic_debug_mode = deterministic_debug_mode
        self.cudnn_enabled = cudnn_enabled

    def setup(self, *args, **kwargs) -> None:
        if self.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self.float32_matmul_precision)

        if self.allow_fp16_reduced_precision_reduction is not None:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                self.allow_fp16_reduced_precision_reduction
            )

        if self.deterministic_debug_mode is not None:
            torch.set_deterministic_debug_mode(self.deterministic_debug_mode)

        if self.cudnn_enabled is not None:
            torch.backends.cudnn.enabled = self.cudnn_enabled
