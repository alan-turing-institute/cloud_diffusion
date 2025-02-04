import numpy as np
import torch
import torchvision.transforms as T

from cloudcasting.constants import IMAGE_SIZE_TUPLE
from cloudcasting.dataset import SatelliteDataset
from cloudcasting.constants import IMAGE_SIZE_TUPLE

def crop_and_uncrop(stride=256, y_start=70, x_start=130):
    y_end = y_start + stride
    x_end = x_start + stride
    
    def crop(img):
        return img[:, :, y_start:y_end, x_start:x_end]

    def uncrop(img):
        blank = np.zeros(IMAGE_SIZE_TUPLE)
        blank = np.where(blank == 0, np.nan, blank)
        blank[:, :, y_start:y_end, x_start:x_end] = img
        return blank
    
    return crop, uncrop


class CloudcastingDataset(SatelliteDataset):
    def __init__(self, stride=256, y_start=70, x_start=130, *args, **kwargs):
        # just to ensure future handling of NaNs works as expected
        if "nan_to_num" in kwargs:
            if kwargs["nan_to_num"]:
                print("nan_to_num must be False for CloudcastingDataset but was set to True; ignoring")
            kwargs.pop("nan_to_num")
        super().__init__(*args, nan_to_num=False, **kwargs)

        self.crop, self.uncrop = crop_and_uncrop(stride, y_start, x_start)

    def __getitem__(self, idx: int):
        # concatenate future prediction and previous frames along time axis
        x, y = super().__getitem__(idx)
        concat_data = np.concatenate((x, y), axis=-3)
 
        # concat_data = np.concatenate((x, y), axis=-3)[self._idxs[idx]]
        # data is in [0,1] range, normalize to [-0.5, 0.5]
        # note that -1s could be NaNs, which are now at +1.5
        # output has shape (11 (if merge_channels is False), history_steps + forecast_horizon, height, width)
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
