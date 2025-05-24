# CT Brain Subcortical Segmentation

The human subcortex contains many structures that hold key responsibilities in many physiological functions fundamental to daily living. These structures have been found to exhibit volume and shape changes during development of neurodegenerative disorders and hence, automated segmentation of these subcortical anatomies will be meaningful for future studies on understanding and developing computer-aided solutions for neurological conditions. The main contributions of this research are:
* Proposal of an automated cross-domain subcortical segmentation label generation framework
* Generation of the first publicly available CT subcortical segmentation dataset
* Training and comparison of various established and SOTA models for CT subcortical segmentation

## Repository Organisation

The code and outputs of this research are organised according to the main contributions in three main folders.

```label_generation_framework``` contains the relevant code for the label generation framework as well as that for consensus voting.

```subcortical_segmentation_dataset``` contains the generated subcortical segmentation masks. The generated subcortical masks contains 17 subcortical regions. The label mapping is available in label_mapping.txt.

```transfer_learning``` contains the notebooks and model weights used for the transfer learning experiments to validate the utility of our generated CT subcortical segmentation dataset.

```segmentation_models_training``` contains the script for running the training of the segmentation models. The models trained include 2D UNet, 3D UNet, Swin UNETR and nnU-Net.

## Set-Up

To run the code in this repository, it is recommended to create a new python/conda virtual environment and install the packages in the requirements.txt file.

### Automated Cross-Domain Label Generation Framework

* ```framework.py``` contains the code necessary to run the both ensembling methods (consensus and majority voting) on the models' outputs to generate new masks.

### Generated Subcortical Segmentation Dataset
* The dataset only contains the generated masks and not the actual CT scans as we do not own the CT scans data and they should be obtained the SynthRAD 2023 Challenge (Task 1) page itself.

* upon obtaining the CT scans, the scans should be processed using the mri_convert tool in the FreeSurfer package. The command is as follows:<br>```mri_convert --conform <input_volume> <output_volume>```

### Transfer Learning
* This folder contains the notebooks used for the transfer learning experiments. The transfer learning experiments were conducted using 3D UNet and ResUNet. The model weights are made available in their respective folders.

### Training of CT Subcortical Segmentation Models
* The models can be found in the ```models``` folder. The UNets are implemented in PyTorch while the SwinUNETR and nnU-Net are imported from MONAI and its python library respectively.

* Training can be ran using ```python model_training.py```.

* Inference can be ran using ```python inference.py```.

* The model and hyperparameters can be adjusted within the training and inference files themselves.

## References

#### Dataset
[IBSR-18: 18 Expert-Annotated MR Dataset](https://www.nitrc.org/projects/ibsr)

[OASIS-TRT-20: 20 Expert-Annotated MR Dataset](https://osf.io/m4svg/)

[SynthRAD 2023 Task 1: 180 MR-CT Paired Dataset](https://synthrad2023.grand-challenge.org/)

#### MR Segmentation Models
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki)

[SAMSEG](https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg)

[FastSurfer](https://github.com/Deep-MI/FastSurfer)

[SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg)

[QuickNAT](https://github.com/ai-med/quickNAT_pytorch)