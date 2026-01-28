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


def brain_extraction(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """ This function assumes freesurfer is set up correctly"""
    tempdir = TemporaryDirectory()
    nii_in_path = Path(tempdir.name) / "in.nii.gz"
    nii_out_path = Path(tempdir.name) / "out.nii.gz"
    nib.save(nii, nii_in_path)
    subprocess.run(["mri_synthstrip", "-i", str(nii_in_path), "-o", str(nii_out_path)], stdout=subprocess.DEVNULL)
    nii = nib_load(nii_out_path, lazy=False)
    return nii
    
    
def get_min_max_nonzero_indices(nii: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    data = nii.get_fdata()
    data_nonzero = np.where(data > 0)
    min_indices = np.min(data_nonzero, axis=1)
    max_indices = np.max(data_nonzero, axis=1)
    return min_indices, max_indices


def get_crop_transform(min_indices: np.ndarray, max_indices: np.ndarray, shape: tuple) -> tio.Crop:
    crop_sides = []
    for i, (low, high) in enumerate(zip(min_indices, max_indices)):
        for e in (low, shape[i]-1-high):
            crop_sides.append(e)
    return tio.Crop(crop_sides)
    

def crop_to_brain(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    min_indices, max_indices = get_min_max_nonzero_indices(nii)
    crop = get_crop_transform(min_indices, max_indices, nii.shape)
    nii = crop(nii)
    return nii
    
    
class EnsureShapeAtLeastTransform(tio.Transform):
    
    def __init__(self, min_shape: tuple):
        super().__init__(parse_input=False)
        self.min_shape = min_shape
        
    def apply_transform(self, nii: nib.Nifti1Image) -> nib.Nifti1Image:
        target_shape = np.max(np.stack([nii.shape, self.min_shape], axis=0), axis=0)
        T = tio.CropOrPad(target_shape=target_shape)
        return T(nii)


class Preprocessor:
    
    def __init__(self, do_bias_field_correction: bool = True, do_brain_extraction: bool = True):
        self.rescale = tio.RescaleIntensity()
        self.resample = tio.Resample(target=1)
        self.to_canonical = tio.ToCanonical()
        self.bias_field_correction = bias_field_correction if do_bias_field_correction else lambda x:x 
        self.brain_extraction = brain_extraction if do_brain_extraction else lambda x:x
        self.crop_to_brain = crop_to_brain
        self.ensure_shape = EnsureShapeAtLeastTransform(min_shape=(160,160,160))
        self.z_score = tio.ZNormalization()
        
    def __call__(self, nii: nib.Nifti1Image) -> nib.Nifti1Image:
        nii = self.rescale(nii)
        nii = self.resample(nii)
        nii = self.to_canonical(nii)
        nii = self.bias_field_correction(nii)
        nii = self.brain_extraction(nii)
        nii = self.crop_to_brain(nii)
        nii = self.ensure_shape(nii)
        nii = self.z_score(nii)
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