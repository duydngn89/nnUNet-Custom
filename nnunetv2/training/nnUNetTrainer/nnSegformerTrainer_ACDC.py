from nnunetv2.training.nnUNetTrainer.nnSegformerTrainer_ACDC_base import (
    ACDCSegformerTrainerBase,
    _make_acdc_stage_mix_trainer,
)

ACDC_ANISOTROPIC_SR_RATIOS = [[1, 4, 4], [1, 2, 2], [1, 1, 1], [1, 1, 1]]


class nnSegformerTrainer_ACDC(ACDCSegformerTrainerBase):
    pass


class nnSegformerTrainer_ACDC_LossDiceCE(ACDCSegformerTrainerBase):
    pass


class nnSegformerTrainer_ACDC_LossDiceCEBoundary(ACDCSegformerTrainerBase):
    multiclass_boundary_weight = 0.5


class nnSegformerTrainer_ACDC_LossDiceCEInverseFreq(ACDCSegformerTrainerBase):
    multiclass_use_inverse_frequency_weights = True


class nnSegformerTrainer_ACDC_LossDiceCEInverseFreqBoundary(ACDCSegformerTrainerBase):
    multiclass_use_inverse_frequency_weights = True
    multiclass_boundary_weight = 0.5


class nnSegformerTrainer_ACDC_Legacy(nnSegformerTrainer_ACDC):
    config_filename = "ACDC_config_legacy.yml"


class nnSegformerTrainer_ACDC_Fixed(nnSegformerTrainer_ACDC):
    config_filename = "ACDC_config_fixed.yml"


nnSegformerTrainer_ACDC_AttentionOnly = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_AttentionOnly",
    ["attention", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)


nnSegformerTrainer_ACDC_AttentionOnly_Legacy = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_AttentionOnly_Legacy",
    ["attention", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_legacy.yml",
)


nnSegformerTrainer_ACDC_AttentionOnly_Fixed = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_AttentionOnly_Fixed",
    ["attention", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_fixed.yml",
)


nnSegformerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention",
    ["mamba", "hybrid", "hybrid", "attention"],
)


nnSegformerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed",
    ["mamba", "hybrid", "hybrid", "attention"],
    config_filename="ACDC_config_fixed.yml",
)


nnSegformerTrainer_ACDC_Stage1Hybrid_Stage234Attention = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage1Hybrid_Stage234Attention",
    ["hybrid", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)


nnSegformerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Legacy = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Legacy",
    ["hybrid", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_legacy.yml",
)


nnSegformerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Fixed = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage1Hybrid_Stage234Attention_Fixed",
    ["hybrid", "attention", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_fixed.yml",
)


nnSegformerTrainer_ACDC_Stage12Hybrid_Stage34Attention = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage12Hybrid_Stage34Attention",
    ["hybrid", "hybrid", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)


nnSegformerTrainer_ACDC_Stage12Hybrid_Stage34Attention_Fixed = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage12Hybrid_Stage34Attention_Fixed",
    ["hybrid", "hybrid", "attention", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
    config_filename="ACDC_config_fixed.yml",
)


nnSegformerTrainer_ACDC_Stage123Hybrid_Stage4Attention = _make_acdc_stage_mix_trainer(
    "nnSegformerTrainer_ACDC_Stage123Hybrid_Stage4Attention",
    ["hybrid", "hybrid", "hybrid", "attention"],
    sr_ratios_override=ACDC_ANISOTROPIC_SR_RATIOS,
)
