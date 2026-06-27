from nnunetv2.training.nnUNetTrainer.nnSegformerTrainer_ACDC_base import (
    ACDCSegformerTrainerBase,
)


class nnSegformerTrainer_ACDC_PaperIso(ACDCSegformerTrainerBase):
    """
    Paper-style isotropic ACDC variant.

    This keeps the shared SegFormer ACDC network/loss behavior but uses the stock
    nnU-Net training transforms inherited from nnUNetTrainer.
    """

    pass
