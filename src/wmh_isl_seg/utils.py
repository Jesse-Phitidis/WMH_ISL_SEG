from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Literal
import numpy as np
import torch
import torch.nn as nn 
from monai.networks.nets import Dynunet
import torchio as tio
from torchio.data.io import sitk_to_nib, nib_to_sitk
import nibabel as nib
import SimpleITK as sitk
from monai.inferers import SlidingWindowInferer


def nib_load(path: Path | str, lazy: bool = True) -> nib.Nifti1Image:
    if lazy:
        return nib.load(path)
    else:
        nii = nib.load(path)
        return nib.Nifti1Image(nii.get_fdata(), nii.affine, nii.header)
    
    
def save_nii(nii: nib.Nifti1Image, path: Path, dtype: str, is_label: bool):
    assert "uint" in dtype, "dtype shoudl be uint8 or uint16"
    bits = int(dtype.split("uint")[-1])
    max_value = (2 ** bits) - 1
    if not is_label:
        nii = tio.RescaleIntensity(out_min_max=(0, max_value))(nii)
    data = nii.get_fdata()
    assert np.min(data) >= 0 and np.max(data) <= max_value
    data = data.astype(dtype)
    nii = nib.Nifti1Image(data, nii.affine, nii.header)
    nii.set_data_dtype(dtype)
    nib.save(nii, path)


def split_join(string: str, delim: str, slc: slice):
    return (delim).join(string.split(delim)[slc])


def bias_field_correction(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    data, affine, header = nii.get_fdata(), nii.affine, nii.header
    sitk_image = nib_to_sitk(data=data[None,...], affine=affine)
    sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1)
    sitk_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_mask)
    data, _ = sitk_to_nib(sitk_image)
    nii = nib.Nifti1Image(data[0,...], affine, header)
    return nii


def brain_extraction(nii: nib.Nifti1Image) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """ This function assumes freesurfer is set up correctly"""
    tempdir = TemporaryDirectory()
    nii_in_path = Path(tempdir.name) / "in.nii.gz"
    nii_out_path = Path(tempdir.name) / "out.nii.gz"
    nii_mask_path = Path(tempdir.name) / "mask.nii.gz"
    nib.save(nii, nii_in_path)
    subprocess.run(
        ["mri_synthstrip", "-i", str(nii_in_path), "-o", str(nii_out_path), "-m", str(nii_mask_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stripped_nii = nib_load(nii_out_path, lazy=False)
    mask_nii = nib_load(nii_mask_path, lazy=False)
    return stripped_nii, mask_nii


def z_score_within_mask(nii: nib.Nifti1Image, mask_nii: nib.Nifti1Image) -> nib.Nifti1Image:
    data = nii.get_fdata()
    mask = mask_nii.get_fdata() > 0.5
    if np.count_nonzero(mask) == 0:
        return nii
    masked_values = data[mask]
    mean = float(np.mean(masked_values))
    std = float(np.std(masked_values))
    if std < 1e-8:
        std = 1.0
    data = (data - mean) / std
    return nib.Nifti1Image(data, nii.affine, nii.header)


class Preprocessor:
    
    def __init__(self, do_bias_field_correction: bool = True):
        self.rescale = tio.RescaleIntensity()
        self.resample = tio.Resample(target=1)
        self.to_canonical = tio.ToCanonical()
        self.bias_field_correction = bias_field_correction if do_bias_field_correction else lambda x:x 
        
    def __call__(self, nii: nib.Nifti1Image, brain_mask: nib.Nifti1Image | None = None) -> nib.Nifti1Image:
        nii = self.rescale(nii)
        nii = self.resample(nii)
        nii = self.to_canonical(nii)
        nii = self.bias_field_correction(nii)

        if brain_mask is None:
            nii, brain_mask = brain_extraction(nii)
        else:
            brain_mask = tio.Resample(target=nii, image_interpolation="nearest")(brain_mask)

        nii = z_score_within_mask(nii, brain_mask)
        return nii
    
    
def build_network():
    net = Dynunet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        kernel_size=[3,3,3,3,3,3],
        strides=[1,2,2,2,2,2],
        upsample_kernel_size=[2,2,2,2,2],
        deep_supervision=True,
        deep_supr_num=3,
        res_block=True
    )
    return net


def load_network(checkpoint: str | Path) -> nn.Module:
    state_dict = torch.load(checkpoint)
    net = build_network()
    net.load_state_dict(state_dict)
    return net


class ModelInferer:
    
    def __init__(self, checkpoints: list, device: str):
        self.checkpoints = checkpoints
        self.device = device
        self.inferer = SlidingWindowInferer(roi_size=(160,160,160), overlap=0.5, mode="gaussian")
        
    def setup(self):
        self.networks = []
        for ckpt in self.checkpoints:
            net = load_network(ckpt)
            net.to(device=self.device)
            net.eval()
            self.networks.append(net)
    
    def clear_memory(self):
        del self.networks
        
    def __call__(self, im: nib.Nifti1Image, binary: bool = False, return_as: Literal["torch", "numpy", "nibabel"] = "nibabel"):
        x = torch.from_numpy(im.get_fdata()).to(dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        preds_lst = []
        with torch.no_grad():
            for net in self.networks:
                pred = self.inferer(x, net)
                pred = torch.softmax(pred, dim=1)
                preds_lst.append(pred)
        preds_batch = torch.cat(preds_lst, dim=0) # use batch dim for different predictions
        pred_ensemble = torch.mean(preds_batch, dim=0).to(device="cpu")
        
        if binary:
            pred_ensemble = torch.argmax(pred_ensemble, dim=0, keepdim=False).to(dtype=torch.int8)
            
        if return_as == "torch":
            return pred_ensemble
        
        if return_as == "numpy":
            return np.array(pred_ensemble)
        
        if return_as == "nibabel":
            return nib.Nifti1Image(np.array(pred_ensemble), affine=im.affine)