Publicly available model for segmentation of White Matter Hyperintensities (WMH) and Ischaemic Stroke Lesions (ISL) from FLAIR MRI. Please cite:

Phitidis, J., Smithard, A. Q., Whiteley, W. N., Wardlaw, J. M., Bernabeu, M. O., & Valdés Hernández, M. (2026). Comparative evaluation of training strategies using partially labelled datasets for segmentation of white matter hyperintensities and stroke lesions in FLAIR MRI. *Artificial intelligence in medicine, 181*, 103507 

# Installation and User Guide

Follow these steps to install and set up the project:

## 1. Create a Conda Environment

Create a new conda environment and install dependencies from `env.yaml`:

```bash
conda env create -f env.yaml
conda activate wmh_isl_seg
```

## 2. Install the Project

Install this project using pip:

```bash
pip install .
```

## 3. Download Checkpoints

Download the checkpoints supplied at this [Google Drive](https://drive.google.com/drive/folders/12JB_pm0SphAOwXzzaFb8reO3Mn-Cj0bT?usp=sharing) link and place them in a directory of your choice. Set the `WMH_ISL_SEG_CHECKPOINTS` environment variable to this directory:

```bash
export WMH_ISL_SEG_CHECKPOINTS=/path/to/checkpoints
```

Replace `/path/to/checkpoints` with the actual path.

## 4. View Prediction Options

Run the following command to see available options:

```bash
wmh_isl_predict --help
```

If the option `--skip_brain_extraction` is not set, then the command `mri_synthstrip` must be set up correctly in your environment.
