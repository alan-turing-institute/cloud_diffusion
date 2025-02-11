import numpy as np
import torch
import torchvision.transforms as T

from cloudcasting.constants import IMAGE_SIZE_TUPLE
from cloudcasting.dataset import SatelliteDataset
from cloudcasting.constants import IMAGE_SIZE_TUPLE

class CloudcastingDataset(SatelliteDataset):
    def __init__(self, stride=256, y_start=70, x_start=130, *args, **kwargs):
        if "nan_to_num" in kwargs:
            if kwargs["nan_to_num"]:
                print("nan_to_num must be False for CloudcastingDataset but was set to True; ignoring")
            kwargs.pop("nan_to_num")
        super().__init__(*args, nan_to_num=False, **kwargs)
        self.stride = stride
        self.y_start = y_start
        self.x_start = x_start
        self.y_end = y_start + stride
        self.x_end = x_start + stride

    def crop(self, img):
        return img[:, :, self.y_start:self.y_end, self.x_start:self.x_end]

    def uncrop(self, img):
        blank = np.zeros(IMAGE_SIZE_TUPLE)
        blank = np.where(blank == 0, np.nan, blank)
        blank[:, :, self.y_start:self.y_end, self.x_start:self.x_end] = img
        return blank

    def __getitem__(self, idx: int):
        x, y = super().__getitem__(idx)
        concat_data = np.concatenate((x, y), axis=-3)
        return self.crop(2*torch.from_numpy(concat_data) - 1)



class DummyNextFrameDataset:
    "Dataset that returns random images"

    def __init__(self, num_frames=4, img_size=64, N=1000):
        self.img_size = img_size
        self.num_frames = num_frames
        self.N = N

    def __getitem__(self, idx):
        return torch.randn(self.num_frames, self.img_size, self.img_size)

    def __len__(self):
        return self.N
