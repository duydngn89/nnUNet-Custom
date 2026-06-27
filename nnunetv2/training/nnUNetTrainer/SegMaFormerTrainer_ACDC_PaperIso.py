from nnunetv2.training.nnUNetTrainer.SegMaFormerTrainer_ACDC_base import (
    ACDCSegMaFormerTrainerBase,
)


class SegMaFormerTrainer_ACDC_PaperIso(ACDCSegMaFormerTrainerBase):
    """
    Paper-style isotropic ACDC variant.

    This keeps the shared SegMaFormer ACDC network/loss behavior but uses the stock
    nnU-Net training transforms inherited from nnUNetTrainer.
    """

    pass
