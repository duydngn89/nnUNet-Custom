# SegMaFormer

SegMaFormer is a custom `nnU-Net v2`-based segmentation framework with stage-wise mixers:

- Stage 1: `mamba`
- Stage 2: `hybrid`
- Stage 3: `hybrid`
- Stage 4: `attention`

This README covers:

- installation
- path configuration
- dataset preprocessing
- model config locations
- how to run the Stage 1 Mamba, Stage 2-3 Hybrid, Stage 4 Attention experiment on BRATS, ACDC, and Synapse/BTCV

## 1. System Requirements

Recommended environment:

- OS: Linux
- Python: `3.11`
- GPU: NVIDIA GPU with CUDA support
- CUDA: `13.x` recommended for this repo
- PyTorch: install a CUDA-enabled build compatible with CUDA 13 before training
- RAM: at least `32 GB`
- VRAM: at least `6 GB` recommended for `3d_fullres`
- Storage: enough space for raw data, preprocessed data, checkpoints, and validation outputs

Practical notes:

- training and preprocessing in this repo are intended for GPU execution
- CPU-only execution is not recommended
- BRATS, ACDC, and BTCV preprocessing can consume substantial disk space in `nnUNet_preprocessed`
- results and checkpoints are written to `nnUNet_results`, so make sure that location has enough free storage

## 2. Installation

Create and activate a Python environment first. A Conda environment is recommended.

```bash
conda create -n segmaformer python=3.11 -y
conda activate segmaformer
```

This repo is currently configured around a CUDA 13 software stack. The pinned dependencies in `requirements.txt`
already include CUDA 13 packages such as `cuda-toolkit==13.0.2`, `cuda-python==13.3.1`,
`nvidia-cudnn-cu13==9.19.0.56`, `nvidia-cusparselt-cu13==0.8.0`, and `nvidia-nccl-cu13==2.28.9`.
Make sure your NVIDIA driver and PyTorch installation are compatible with CUDA 13 before continuing.

Install a CUDA 13-compatible PyTorch build, then install the project:

```bash
cd nnUNet-Custom
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation mamba-ssm
```

Optional CUDA sanity check:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available())"
```

Check that the nnU-Net commands are available:

```bash
nnUNetv2_train -h
nnUNetv2_plan_and_preprocess -h
```

## 3. Configure nnU-Net Paths

Set the three required nnU-Net environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

Example:

```bash
export nnUNet_raw="/mnt/new_volume/nnUnetFramwork-IntegratedCustomModel/nnUNet_raw"
export nnUNet_preprocessed="/mnt/new_volume/nnUnetFramwork-IntegratedCustomModel/nnUNet_preprocessed"
export nnUNet_results="/mnt/new_volume/nnUnetFramwork-IntegratedCustomModel/nnUNet_results"
```

You can place these in `~/.bashrc` so they persist across sessions.

## 4. Dataset Layout

This repo expects standard `nnU-Net v2` dataset structure:

- `nnUNet_raw/Dataset042_BraTS2017`
- `nnUNet_raw/Dataset027_ACDC`
- `nnUNet_raw/Dataset001_BTCV`

In this workspace, the dataset IDs used by the training commands are:

- BRATS: `42`
- ACDC: `27`
- Synapse/BTCV: `1`

If your dataset IDs are different, update the commands accordingly.

## 5. Data Conversion

Before preprocessing, each dataset must be arranged in `nnU-Net v2` raw format:

```bash
nnUNet_raw/DatasetXXX_NAME/
├── imagesTr
├── labelsTr
├── imagesTs
└── dataset.json
```

### ACDC

This repo includes an ACDC conversion script:

- [nnunetv2/dataset_conversion/Dataset027_ACDC.py]
Expected source layout:

- input folder contains `training/` and `testing/`
- each patient folder contains `.nii.gz` images and `_gt.nii.gz` labels

Run:

```bash
python nnunetv2/dataset_conversion/Dataset027_ACDC.py \
  -i /path/to/ACDC \
  -d 27
```

What it does:

- copies training images to `imagesTr`
- copies training labels to `labelsTr`
- copies testing images to `imagesTs`
- creates `dataset.json`
- creates `splits_final.json` in `nnUNet_preprocessed/Dataset027_ACDC`

### BRATS

This repo includes a BraTS conversion script:

- [nnunetv2/dataset_conversion/Dataset042_BraTS18.py]

Important:

- this script is project-specific
- it expects `.pt` files, not the standard original BraTS NIfTI release
- it reads:
  - `*_modalities.pt`
  - `*_label.pt`

It converts:

- 4 input modalities into nnU-Net channels
- one-hot tumor labels into nnU-Net region labels

Before using it, update the source path inside the script:

- `pt_root = "/your/path/to/BraTS2017_Training_Data"`

Then run:

```bash
python nnunetv2/dataset_conversion/Dataset042_BraTS18.py
```

If your BraTS data is already available in `nnUNet_raw/Dataset042_BraTS2017`, you do not need to run this conversion again.

### Synapse / BTCV

In this workspace, BTCV is already prepared as:

- `nnUNet_raw/Dataset001_BTCV`

The existing [dataset.json] shows:

- single-channel CT input
- 13 abdominal organ labels
- `file_ending: .nii.gz`

For BTCV/Synapse, the repo currently assumes you already have the raw folder prepared in nnU-Net format. That means:

- training images in `imagesTr`
- training labels in `labelsTr`
- optional test images in `imagesTs`
- a valid `dataset.json`

If your BTCV/Synapse data is still in another format, convert it manually into nnU-Net raw structure first, then continue with preprocessing.

## 6. Preprocess the Datasets

Run planning and preprocessing once per dataset:

```bash
nnUNetv2_plan_and_preprocess -d 42 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 27 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 1  --verify_dataset_integrity
```

## 7. Model Configuration

The architecture YAML files are here:

- BRATS: [nnunetv2/training/nnUNetTrainer/network_architecture/configs/BRATS_config.yml]
- ACDC fixed: [nnunetv2/training/nnUNetTrainer/network_architecture/configs/ACDC_config_fixed.yml]
- Synapse/BTCV fixed: [nnunetv2/training/nnUNetTrainer/network_architecture/configs/BCTV_config_fixed.yml]

What to edit in the YAML files:

- `embed_dims`
- `patch_kernel_size`
- `patch_stride`
- `patch_padding`
- `mlp_ratios`
- `num_heads`
- `depths`
- `d_state`
- `d_conv`
- `expand`
- `decoder_head_embedding_dim`
- `deep_supervision`

What controls the stage mixers:

- the stage layout is selected by the trainer class, not by the YAML alone
- for the experiment in this README, use the dedicated trainer classes below

## 8. Trainer Names for the Target Experiment

The Stage 1 Mamba, Stage 2-3 Hybrid, Stage 4 Attention setup uses:

- BRATS: `SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention`
- ACDC: `SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed`
- Synapse/BTCV: `SegMaFormerTrainer_BTCV_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed`

Notes:

- BRATS uses `BRATS_config.yml`
- ACDC fixed uses `ACDC_config_fixed.yml`
- Synapse/BTCV fixed uses `BCTV_config_fixed.yml`
- in the current codebase, SR is only retained for attention stages

## 9. Run the Experiments

All commands below use:

- configuration: `3d_fullres`
- fold: `0`
- plans: `nnUNetPlans`
- single GPU: `-num_gpus 1`

### BRATS

```bash
nnUNetv2_train 42 3d_fullres 0 \
  -tr SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention \
  -p nnUNetPlans \
  -num_gpus 1
```

### ACDC

```bash
nnUNetv2_train 27 3d_fullres 0 \
  -tr SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed \
  -p nnUNetPlans \
  -num_gpus 1
```

### Synapse / BTCV

```bash
nnUNetv2_train 1 3d_fullres 0 \
  -tr SegMaFormerTrainer_BTCV_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed \
  -p nnUNetPlans \
  -num_gpus 1
```

If you want to choose a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 42 3d_fullres 0 \
  -tr SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention \
  -p nnUNetPlans \
  -num_gpus 1
```

## 10. Output Locations

Training outputs are written to:

```bash
${nnUNet_results}/DatasetXXX_NAME/TRAINER__PLANS__3d_fullres/
```

Examples:

```bash
nnUNet_results/Dataset042_BraTS2017/SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention__nnUNetPlans__3d_fullres/

nnUNet_results/Dataset027_ACDC/SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed__nnUNetPlans__3d_fullres/

nnUNet_results/Dataset001_BTCV/SegMaFormerTrainer_BTCV_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed__nnUNetPlans__3d_fullres/
```

Useful files:

- `training_log_*.txt`
- `checkpoint_latest.pth`
- `checkpoint_best.pth`
- `validation/summary.json`

## 11. Validation / Inference

Run validation during training with `--val` if needed:

```bash
nnUNetv2_train 42 3d_fullres 0 \
  -tr SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention \
  -p nnUNetPlans \
  -num_gpus 1 \
  --val
```

After training, check:

- `validation/summary.json`
- predicted segmentations under the trainer output folder

## 12. Quick Summary

For the main experiment in this repo:

- BRATS:
  `SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention`
- ACDC:
  `SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed`
- Synapse/BTCV:
  `SegMaFormerTrainer_BTCV_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed`

Core commands:

```bash
nnUNetv2_plan_and_preprocess -d 42 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 27 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 1  --verify_dataset_integrity

nnUNetv2_train 42 3d_fullres 0 -tr SegMaFormerTrainer_BRATS_Stage1Mamba_Stage23Hybrid_Stage4Attention -p nnUNetPlans -num_gpus 1
nnUNetv2_train 27 3d_fullres 0 -tr SegMaFormerTrainer_ACDC_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed -p nnUNetPlans -num_gpus 1
nnUNetv2_train 1  3d_fullres 0 -tr SegMaFormerTrainer_BTCV_Stage1Mamba_Stage23Hybrid_Stage4Attention_Fixed -p nnUNetPlans -num_gpus 1
```
