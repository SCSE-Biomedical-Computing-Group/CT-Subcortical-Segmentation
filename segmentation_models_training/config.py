from models.UNet2D import UNet2D
from models.UNet3D import UNet3D
from models.SwinUNETR import SwinUNETR
from Dataset import Dataset2D, Dataset3D

MODEL_CONFIG = {
    "UNet2D": UNet2D, "UNet3D": UNet3D, "SwinUNETR": SwinUNETR
}

DATASET_CONFIG = { "2D": Dataset2D, "3D": Dataset3D }