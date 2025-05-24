import nibabel as nib
import numpy as np
import copy
import os
import glob

def filter_labels(input_mask):
    filtered_labels = [4, 43, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58, 16]

    mask_data = input_mask.get_fdata()
    filtered_mask_data = np.isin(mask_data, filtered_labels) * mask_data
    filtered_mask = nib.Nifti1Image(filtered_mask_data, input_mask.affine, input_mask.header)

    return filtered_mask

def rename_labels(input_mask_path):
    # Label map for QuickNAT model output
    label_map = {3: 4, 20: 43, 7: 10, 24: 49, 8: 11, 25: 50, 9: 12, 26: 51, 10: 13, 27: 52, 14: 17, 28: 53, 15: 18, 29: 54, 16: 26, 30: 58, 13: 16}
    
    # Label map for OASIS-TRT-20 Dataset
    # label_map = {17: 4, 183: 43, 43: 10, 209: 49, 47: 11, 213: 50, 51: 12, 217: 51, 55: 13, 221: 52, 72: 17, 226: 53, 77: 18, 230: 54, 111: 26, 247: 58, 68: 16}

    mask_img = nib.load(input_mask_path)
    mask_data = mask_img.get_fdata().astype(int)
    renamed_mask_data = copy.deepcopy(mask_data)

    for old_label, new_label in label_map.items():
        renamed_mask_data[mask_data == new_label] = label_map.get(new_label, 0)
        renamed_mask_data[mask_data == old_label] = new_label

    renamed_mask = nib.Nifti1Image(renamed_mask_data, mask_img.affine, mask_img.header)

    return renamed_mask

def majority_voting(input_masks, weights, output_mask_path):
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    masks_data = [mask.get_fdata().astype(int) for mask in input_masks]
    masks_combined = np.stack(masks_data, axis=0)

    classes = np.unique(masks_combined)
    voxel_votes = np.zeros(masks_combined.shape[1:] + (len(classes),), dtype=float)
    for idx, label in enumerate(classes):
        num_votes_for_label = np.sum((masks_combined == label) * weights[:, None, None, None], axis=0)
        voxel_votes[..., idx] = num_votes_for_label

    max_class_votes = np.max(voxel_votes, axis=-1)
    max_class_tied = (voxel_votes == max_class_votes[..., None]).sum(axis=-1) > 1
    new_labels = np.zeros_like(max_class_votes, dtype=int)
    for idx, label in enumerate(classes):
        tied_voxels = max_class_tied & (voxel_votes[..., idx] == max_class_votes)
        not_tied_voxels = ~max_class_tied & (voxel_votes[..., idx] == max_class_votes)

        new_labels[tied_voxels] = 0
        new_labels[not_tied_voxels] = label

    new_mask = nib.Nifti1Image(new_labels, input_masks[0].affine, input_masks[0].header)
    nib.save(new_mask, output_mask_path)

def consensus_voting(input_masks, output_path):
    masks = np.stack([m.get_fdata().astype(int) for m in input_masks], axis=0)
    consensus = np.where(np.all(masks == masks[0], axis=0), masks[0], 0)
    new_mask = nib.Nifti1Image(consensus, affine=input_masks[0].affine, header=input_masks[0].header)
    nib.save(new_mask, output_path)


if __name__ == "__main__":
    mask_folder = ""
    mask_names = sorted(glob.glob(os.path.join(mask_folder, "*.nii.gz")))
    mask_names = [mask.split('/')[-1].split('.')[0] for mask in mask_names]
    
    weights = [1, 1, 1]

    for mask in mask_names:
        fastsurfer_img = f"data/fastsurfer/{mask}/mri/aseg.nii.gz"
        synthseg_img = f"data/synthseg/{mask}_ana.nii.gz"
        freesurfer_img = f"data/freesurfer/{mask}.nii.gz"
        samseg_img = f"data/samseg/{mask}_seg.nii.gz"
        quicknat_img = f"data/quicknat/{mask}_ana_processed.nii"

        # Ensemble all models
        # input_masks = [nib.load(freesurfer_img), nib.load(samseg_img), nib.load(fastsurfer_img), nib.load(synthseg_img), rename_labels(quicknat_img)]

        # Ensemble DL models only
        input_masks = [nib.load(fastsurfer_img), nib.load(synthseg_img), rename_labels(quicknat_img)]


        input_masks = [filter_labels(input_mask) for input_mask in input_masks]

        # Majority Voting
        # majority_voting(input_masks, weights, f"IBSR_masks/mv/{mask}.nii.gz")
        
        # Consensus Voting
        consensus_voting(input_masks, f"IBSR_masks/consensus/{mask}")