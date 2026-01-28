import argparse
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import torchio as tio
import nibabel as nib
from wmh_isl_seg.utils import Preprocessor, ModelInferer, save_nii


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", type=str, required=True, help="Path or csv of paths to FLAIR nifti files")
    parser.add_argument("-o", "--outputs", type=str, required=True, help="Path or csv of paths to save the output files")
    parser.add_argument("--skip_bias_field_correction", action="store_true", default=False, help="Skip applying bias field correction")
    parser.add_argument("--skip_brain_extraction", action="store_true", default=False, help="Skip applying brain extraction (requires mri_synthstrip to be set up)")
    parser.add_argument("--cpu", action="store_true", default=False, help="Run on the CPU")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    # Load and check file paths
    if args.inputs.endswith(".csv") or args.outputs.endswith(".csv"):
        assert args.inputs.endswith(".csv") and args.outputs.endswith(".csv"), "Both inputs and outputs must be csv files if one is a csv"
        inputs = [Path(x) for x in pd.read_csv(args.inputs, header=None).values.flatten().tolist()]
        outputs = [Path(x) for x in pd.read_csv(args.outputs, header=None).values.flatten().tolist()]
        assert len(inputs) == len(outputs), "Inputs and outputs csv files must have the same number of entries"
    elif args.inputs.endswith(".nii.gz") or args.outputs.endswith(".nii.gz"):
        assert args.inputs.endswith(".nii.gz") and args.outputs.endswith(".nii.gz"), "Both inputs and outputs must be nifti files if one is a nifti"
        inputs = [Path(args.inputs)]
        outputs = [Path(args.outputs)]
    else:
        raise ValueError("inputs and outputs must have extension '.csv' or '.nii.gz'")
    
    # Check cuda is available 
    if not args.cpu:
        assert torch.cuda.is_available(), "CUDA is not available. Use --cpu to run on CPU."
    
    # Get checkpoints
    checkpoints = sorted(Path(os.environ.get("WMH_ISL_SEG_CHECKPOINTS")).glob("*.ckpt"))
    
    # Set up preprocessor
    preprocessor = Preprocessor(
        do_bias_field_correction = (not args.skip_bias_field_correction), 
        do_brain_extraction = (not args.skip_brain_extraction)
        )
    
    # Set up ensemble inferer
    inferer = ModelInferer(
        checkpoints = checkpoints,
        device = "cuda" if not args.cpu else "cpu"
    )
    inferer.setup()
    
    # Predict
    for in_path, out_path in tqdm(zip(inputs, outputs), total=len(inputs)):
        nii_original = nib.load(in_path)
        nii_preprocessed = preprocessor(nii_original)
        nii_prediction = inferer(nii_preprocessed, binary=True)
        nii_prediction = tio.Resample(target=(nii_original.shape, nii_original.affine), image_interpolation="nearest")(nii_prediction)
        save_nii(
            nii=nii_prediction,
            path=out_path,
            dtype="uint8",
            is_label=True
        )


if __name__ == "__main__":
    main()