from nnunetv2.training.nnUNetTrainer.SegMaFormerTrainer_ACDC_base import (
    _make_acdc_stage_mix_trainer,
)


SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_PaperIso = _make_acdc_stage_mix_trainer(
    "SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_PaperIso",
    ["mamba", "hybrid", "hybrid", "attention"],
)
