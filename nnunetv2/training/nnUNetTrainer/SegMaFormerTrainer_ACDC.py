from nnunetv2.training.nnUNetTrainer.SegMaFormerTrainer_ACDC_base import (
    ACDCSegMaFormerTrainerBase,
    _make_acdc_stage_mix_trainer,
)

ACDC_ANISOTROPIC_SR_RATIOS = [[1, 4, 4], [1, 2, 2], [1, 1, 1], [1, 1, 1]]


class SegMaFormerTrainer_ACDC(ACDCSegMaFormerTrainerBase):
    pass


class SegMaFormerTrainer_ACDC_RoPEOff(SegMaFormerTrainer_ACDC):
    rope_enabled = False


class SegMaFormerTrainer_ACDC_LossDiceCE(ACDCSegMaFormerTrainerBase):
    pass


class SegMaFormerTrainer_ACDC_LossDiceCEBoundary(ACDCSegMaFormerTrainerBase):
    multiclass_boundary_weight = 0.5


class SegMaFormerTrainer_ACDC_LossDiceCEInverseFreq(ACDCSegMaFormerTrainerBase):
    multiclass_use_inverse_frequency_weights = True


class SegMaFormerTrainer_ACDC_LossDiceCEInverseFreqBoundary(ACDCSegMaFormerTrainerBase):
    multiclass_use_inverse_frequency_weights = True
    multiclass_boundary_weight = 0.5


class SegMaFormerTrainer_ACDC_Legacy(SegMaFormerTrainer_ACDC):
    config_filename = "ACDC_config_legacy.yml"


class SegMaFormerTrainer_ACDC_Fixed(SegMaFormerTrainer_ACDC):
    config_filename = "ACDC_config_fixed.yml"


SegMaFormerTrainer_ACDC_AttentionOnly = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_AttentionOnly",
    ["attention", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)


SegMaFormerTrainer_ACDC_AttentionOnly_Legacy = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_AttentionOnly_Legacy",
    ["attention", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_legacy.yml",
)


SegMaFormerTrainer_ACDC_AttentionOnly_Fixed = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_AttentionOnly_Fixed",
    ["attention", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_fixed.yml",
)


# Canonical SegFormer3D baseline: all encoder stages use attention with
# ACDC's fixed patch-stride schedule.
SegMaFormerTrainer_ACDC_SegFormer3D_FixedStride = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_SegFormer3D_FixedStride",
    ["attention", "attention", "attention", "attention"],
    config_filename="ACDC_config_Segformer3D_fixed.yml",
)


SegMaFormerTrainer_ACDC_SegFormer3D_FixedStride_RoPEOff = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_SegFormer3D_FixedStride_RoPEOff",
    ["attention", "attention", "attention", "attention"],
    config_filename="ACDC_config_Segformer3D_fixed.yml",
    rope_enabled=False,
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention",
    ["mamba", "hybrid", "hybrid", "attention"],
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed",
    ["mamba", "hybrid", "hybrid", "attention"],
    config_filename="ACDC_config_fixed.yml",
)

SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed_GateOff = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed_GateOff",
    ["mamba", "hybrid", "hybrid", "attention"],
    config_filename="ACDC_config_fixed.yml",
    disable_hybrid_gate=True,
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage234Attention_Fixed = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage234Attention_Fixed",
    ["mamba", "attention", "attention", "attention"],
    config_filename="ACDC_config_fixed.yml",
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_RoPEOff = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_RoPEOff",
    ["mamba", "hybrid", "hybrid", "attention"],
    rope_enabled=False,
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed_RoPEOff = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed_RoPEOff",
    ["mamba", "hybrid", "hybrid", "attention"],
    config_filename="ACDC_config_fixed.yml",
    rope_enabled=False,
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage234Attention_Fixed_RoPEOff = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage234Attention_Fixed_RoPEOff",
    ["mamba", "attention", "attention", "attention"],
    config_filename="ACDC_config_fixed.yml",
    rope_enabled=False,
)


SegMaFormerTrainer_ACDC_Stage1Hybrid_Stage234Attention = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Hybrid_Stage234Attention",
    ["hybrid", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)


SegMaFormerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Legacy = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Legacy",
    ["hybrid", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_legacy.yml",
)


SegMaFormerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Fixed = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Fixed",
    ["hybrid", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_fixed.yml",
)


SegMaFormerTrainer_ACDC_Stage12Hybrid_Stage34Attention = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage12Hybrid_Stage34Attention",
    ["hybrid", "hybrid", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)


SegMaFormerTrainer_ACDC_Stage12Hybrid_Stage34Attention_Fixed = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage12Hybrid_Stage34Attention_Fixed",
    ["hybrid", "hybrid", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_fixed.yml",
)


SegMaFormerTrainer_ACDC_Stage123Mamba_Stage4Attention = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage123Mamba_Stage4Attention",
    ["mamba", "mamba", "mamba", "attention"],
)


SegMaFormerTrainer_ACDC_Stage123Mamba_Stage4Attention_Fixed = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage123Mamba_Stage4Attention_Fixed",
    ["mamba", "mamba", "mamba", "attention"],
    config_filename="ACDC_config_fixed.yml",
)


SegMaFormerTrainer_ACDC_Stage123Hybrid_Stage4Attention = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage123Hybrid_Stage4Attention",
    ["hybrid", "hybrid", "hybrid", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)
