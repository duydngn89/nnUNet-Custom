"""nnU-Net v2 trainer for the official nnFormer architecture.

The network source is vendored under ``network_architecture/nnFormer.py`` from
https://github.com/282857341/nnFormer (MIT). This trainer intentionally keeps
nnU-Net v2 preprocessing, folds, augmentation, and evaluation unchanged so it
can be used as a fair architecture baseline in this repository.
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.nnUNetTrainer.network_architecture.nnFormer import nnFormer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _PlanPaddedNNFormer(nn.Module):
    """Run fixed-grid nnFormer on a padded plan patch and restore its logits."""

    def __init__(self, network: nn.Module, planned_patch_size: Sequence[int]):
        super().__init__()
        self.network = network
        self.planned_patch_size = tuple(int(value) for value in planned_patch_size)

    def forward(self, x: torch.Tensor):
        original_shape = tuple(x.shape[2:])
        padding = []
        for actual, planned in zip(reversed(original_shape), reversed(self.planned_patch_size)):
            padding.extend((0, planned - actual))
        x = F.pad(x, padding)
        logits = self.network(x)
        outputs = logits if isinstance(logits, (list, tuple)) else [logits]
        cropped = []
        for level, output in enumerate(outputs):
            target_shape = tuple(max(1, size // (2 ** level)) for size in original_shape)
            cropped.append(output[(...,) + tuple(slice(0, size) for size in target_shape)])
        return cropped if isinstance(logits, (list, tuple)) else cropped[0]


class NNFormerTrainer(nnUNetTrainer):
    """Train nnFormer with the active nnU-Net v2 plan and fold.

    This is the official Synapse/anisotropic nnFormer variant: a [2, 4, 4]
    patch embedding, 192 channels, depths [2, 2, 2, 2], and local/global
    volume attention with skip attention. The fixed attention grid is padded
    to a compatible shape internally, then logits are cropped to the plan.
    """

    embedding_dim = 192
    depths = (2, 2, 2, 2)
    num_heads = (6, 12, 24, 48)
    embedding_patch_size = (2, 4, 4)
    window_size = (4, 4, 8, 4)

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True

    def _get_deep_supervision_scales(self):
        # nnFormer emits logits at full, half, and quarter resolution. The
        # trailing zero-weight target gives the original 1 : 1/2 : 1/4 loss
        # weighting while keeping nnU-Net's augmentation interface unchanged.
        return [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                [0.125, 0.125, 0.125]]

    def set_deep_supervision_enabled(self, enabled: bool):
        """Toggle nnFormer's native output heads through the padding wrapper."""
        module = self.network.module if self.is_ddp else self.network
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        if not isinstance(module, _PlanPaddedNNFormer):
            raise TypeError(f"Expected _PlanPaddedNNFormer, got {type(module).__name__}")
        module.network.do_ds = enabled
        module.network._deep_supervision = enabled

    @staticmethod
    def _compatible_patch_size(patch_size: Sequence[int]) -> tuple[int, int, int]:
        required_multiple = (16, 32, 32)
        if len(patch_size) != 3 or any(size < 1 for size in patch_size):
            raise ValueError(
                f"NNFormerTrainer requires a positive three-dimensional patch size, got {tuple(patch_size)}"
            )
        return tuple(
            ((int(size) + multiple - 1) // multiple) * multiple
            for size, multiple in zip(patch_size, required_multiple)
        )

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Sequence[str],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        patch_size = tuple(int(value) for value in self.configuration_manager.patch_size)
        compatible_patch_size = self._compatible_patch_size(patch_size)
        if compatible_patch_size != patch_size:
            self.print_to_log_file(
                f"NNFormer pads plan patch {patch_size} to fixed attention grid {compatible_patch_size}"
            )
        network = nnFormer(
            crop_size=compatible_patch_size,
            embedding_dim=self.embedding_dim,
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            conv_op=nn.Conv3d,
            depths=list(self.depths),
            num_heads=list(self.num_heads),
            patch_size=list(self.embedding_patch_size),
            window_size=list(self.window_size),
            deep_supervision=enable_deep_supervision,
        )
        return _PlanPaddedNNFormer(network, compatible_patch_size)
