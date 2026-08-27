import os

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import (
    DC_and_BCE_loss,
    DC_and_CE_loss,
    DC_and_CE_and_Boundary_loss,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.network_architecture.SegMaFormer import (
    build_segmaformer_model_from_yaml,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class ACDCSegMaFormerTrainerBase(nnUNetTrainer):
    config_filename = "ACDC_config.yml"
    deep_supervision_weights = np.array([1.0, 0.25, 0.125, 0.125, 0.125], dtype=np.float32)
    block_mode = "hybrid"
    stage_block_types = None
    rope_enabled = True
    disable_hybrid_gate = False
    sr_ratios_override = None
    num_mamba_replacements = None
    replacement_block_mode = "mamba"
    multiclass_use_inverse_frequency_weights = False
    multiclass_boundary_weight = 0.0
    multiclass_boundary_dilation = 3

    @classmethod
    def build_network_architecture(
        cls,
        network_arch_class_name,
        network_arch_init_kwargs,
        network_arch_init_kwargs_req_import,
        num_input_channels,
        num_classes,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if not isinstance(num_input_channels, int):
            raise ValueError(f"Invalid num_input_channels: {num_input_channels}")

        config_path = os.path.join(
            os.path.dirname(__file__),
            "network_architecture",
            "configs",
            cls.config_filename,
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SegMaFormer config not found: {config_path}")

        print(f"[SegMaFormer] Loading config from: {config_path}")
        print(f"[SegMaFormer] ACDC ablation mode: {cls.block_mode}")
        if cls.stage_block_types is not None:
            print(f"[SegMaFormer] Explicit stage block types: {cls.stage_block_types}")
        print(f"[SegMaFormer] RoPE enabled: {cls.rope_enabled}")
        print(f"[SegMaFormer] Hybrid gate enabled: {not cls.disable_hybrid_gate}")
        if cls.sr_ratios_override is not None:
            print(f"[SegMaFormer] Attention sr_ratios override: {cls.sr_ratios_override}")
        if cls.num_mamba_replacements is not None:
            print(
                f"[SegMaFormer] Attention->Mamba replacements: {cls.num_mamba_replacements} "
                f"using mode={cls.replacement_block_mode}"
            )

        overrides = {
            "block_mode": cls.block_mode,
            "stage_block_types": cls.stage_block_types,
            "use_rope": cls.rope_enabled,
            "disable_hybrid_gate": cls.disable_hybrid_gate,
            "sr_ratios": cls.sr_ratios_override,
            "num_mamba_replacements": cls.num_mamba_replacements,
            "replacement_block_mode": cls.replacement_block_mode,
        }
        overrides = {key: value for key, value in overrides.items() if value is not None}

        model = build_segmaformer_model_from_yaml(
            config_path,
            overrides=overrides,
        )

        if hasattr(model, "deep_supervision") and model.deep_supervision != enable_deep_supervision:
            print(
                f"[SegMaFormer] Overriding deep_supervision "
                f"from {model.deep_supervision} to {enable_deep_supervision}"
            )
            model.deep_supervision = enable_deep_supervision

        print(f"[SegMaFormer] Model ready | Inputs: {num_input_channels} | Outputs: {num_classes}")
        return model

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                weight_ce=0.2,
                weight_dice=0.8,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = self._build_multiclass_loss()

        if self._do_i_compile():
            try:
                loss = torch.compile(loss)
                print("[SegMaFormer] Compiled loss with torch.compile()")
            except Exception as exc:
                print(f"[SegMaFormer] torch.compile() failed for loss: {exc}")

        if self.enable_deep_supervision:
            weights = self.deep_supervision_weights.astype(np.float32, copy=True)
            weights /= weights.sum()
            print(f"[SegMaFormer] Deep supervision weights: {weights.tolist()}")
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _build_multiclass_loss(self):
        if (
            not self.multiclass_use_inverse_frequency_weights
            and self.multiclass_boundary_weight == 0
        ):
            return DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1.0,
                weight_dice=1.0,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        return DC_and_CE_and_Boundary_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": True,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_boundary=self.multiclass_boundary_weight,
            ignore_label=self.label_manager.ignore_label,
            boundary_dilation=self.multiclass_boundary_dilation,
            use_class_weights=self.multiclass_use_inverse_frequency_weights,
        )


def _make_acdc_stage_mix_trainer(class_name: str, stage_block_types, **attrs):
    namespace = {"stage_block_types": list(stage_block_types)}
    namespace.update(attrs)
    return type(class_name, (ACDCSegMaFormerTrainerBase,), namespace)
