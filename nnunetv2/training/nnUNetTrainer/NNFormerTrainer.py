"""nnU-Net v2 trainer for the official nnFormer architecture.

The network source is vendored under ``network_architecture/nnFormer.py`` from
https://github.com/282857341/nnFormer (MIT). This trainer intentionally keeps
nnU-Net v2 preprocessing, folds, augmentation, and evaluation unchanged so it
can be used as a fair architecture baseline in this repository.
"""

from typing import Sequence

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.network_architecture.nnFormer import nnFormer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class NNFormerTrainer(nnUNetTrainer):
    """Train nnFormer with the active nnU-Net v2 plan and fold.

    This is the official Synapse/anisotropic nnFormer variant: a [2, 4, 4]
    patch embedding, 192 channels, depths [2, 2, 2, 2], and local/global
    volume attention with skip attention. Its full-resolution prediction head
    and its three official deep-supervision outputs are retained.
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

    @staticmethod
    def _validate_patch_size(patch_size: Sequence[int]) -> None:
        required_multiple = (16, 32, 32)
        if len(patch_size) != 3 or any(size < multiple or size % multiple
                                       for size, multiple in zip(patch_size, required_multiple)):
            raise ValueError(
                "NNFormerTrainer requires the 3D plan patch size to be divisible by "
                f"{required_multiple}; got {tuple(patch_size)}. Create an nnFormer-specific "
                "plan (or adjust its patch size) before training."
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
        self._validate_patch_size(patch_size)
        return nnFormer(
            crop_size=patch_size,
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
