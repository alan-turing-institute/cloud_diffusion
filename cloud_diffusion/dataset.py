from pathlib import Path

import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as T
import wandb
from fastprogress import progress_bar

from cloudcasting.constants import IMAGE_SIZE_TUPLE, NUM_CHANNELS
from cloudcasting.dataset import SatelliteDataset

from cloud_diffusion.utils import ls

PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = "capecape/gtc/np_dataset:v0"
from cloudcasting.constants import IMAGE_SIZE_TUPLE

def crop_and_uncrop(stride=256, y_start=70,  x_start=130,):
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
    def __init__(self, img_size, valid=False, strategy=None, merge_channels=False, *args, **kwargs):
        
        if "nan_to_num" in kwargs:
            if kwargs["nan_to_num"]:
                print("nan_to_num must be False for CloudcastingDataset but was set to True; ignoring")
            kwargs.pop("nan_to_num")
        super().__init__(*args, nan_to_num=False, **kwargs)
        if strategy == "resize":
            tfms = [T.Resize((img_size, int(img_size * (IMAGE_SIZE_TUPLE[1] / IMAGE_SIZE_TUPLE[0]))))] if img_size is not None else []
            tfms += [T.RandomCrop(img_size)] if not valid else [T.CenterCrop(img_size)]
        elif strategy == "centercrop":
            tfms = [T.CenterCrop(img_size)]
        elif strategy is None:
            tfms = []
        else:
            raise ValueError(f"Strategy {strategy} not found")
        self.tfms = T.Compose(tfms)
        self.merge_channels = merge_channels
        self.crop, self.uncrop = crop_and_uncrop()

        # if merge_channels:
        #     # for each entry in the dataset, randomly select a channel to keep.
        #     # note this is deterministic for every entry in the dataset;
        #     # you will get the same set of channels every epoch.
        #     self._idxs = npr.choice(NUM_CHANNELS, size=super().__len__(), replace=True)
        # else:
        #     self._idxs = [...] * super().__len__()  # x[...] just returns x


    def __getitem__(self, idx: int):
        # concatenate future prediction and previous frames along time axis
        x, y = super().__getitem__(idx)
        concat_data = np.concatenate((x, y), axis=-3)
 
        # concat_data = np.concatenate((x, y), axis=-3)[self._idxs[idx]]
        # data is in [0,1] range, normalize to [-0.5, 0.5]
        # note that -1s could be NaNs, which are now at +1.5
        # output has shape (11 (if merge_channels is False), history_steps + forecast_horizon, height, width)
        return self.crop(2*self.tfms(torch.from_numpy(concat_data)) - 1)



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


class CloudDataset:
    """Dataset for cloud images
    It loads numpy files from wandb artifact and stacks them into a single array
    It also applies some transformations to the images
    """

    def __init__(
        self,
        files,  # list of numpy files to load (they come from the artifact)
        num_frames=4,  # how many consecutive frames to stack
        scale=True,  # if we images to interval [-0.5, 0.5]
        img_size=64,  # resize dim, original images are big (446, 780)
        valid=False,  # if True, transforms are deterministic
    ):
        tfms = [T.Resize((img_size, int(img_size * 1.7)))] if img_size is not None else []
        tfms += [T.RandomCrop(img_size)] if not valid else [T.CenterCrop(img_size)]
        self.tfms = T.Compose(tfms)
        self.load_data(files, num_frames, scale)

    def load_day(self, file, scale=True):
        one_day = np.load(file)
        if scale:
            one_day = 0.5 - self._scale(one_day)
        return one_day

    def load_data(self, files, num_frames, scale):
        "Loads all data into a single array self.data"
        data = []
        for file in (pbar := progress_bar(files, leave=False)):
            one_day = self.load_day(file, scale)
            wds = np.lib.stride_tricks.sliding_window_view(one_day.squeeze(), num_frames, axis=0).transpose((0, 3, 1, 2))
            data.append(wds)
            pbar.comment = f"Creating CloudDataset from {file}"
        self.data = np.concatenate(data, axis=0)

    def shuffle(self):
        """Shuffles the dataset, useful for getting
        interesting samples on the validation dataset"""
        idxs = torch.randperm(len(self.data))
        self.data = self.data[idxs]
        return self

    @staticmethod
    def _scale(arr):
        "Scales values of array in [0,1]"
        m, M = arr.min(), arr.max()
        return (arr - m) / (M - m)

    def __getitem__(self, idx):
        return self.tfms(torch.from_numpy(self.data[idx]))

    def __len__(self):
        return len(self.data)

    def save(self, fname="cloud_frames.npy"):
        np.save(fname, self.data)


class CloudDatasetInference(CloudDataset):
    def load_data(self, files, num_frames=None, scale=None):
        "Loads all data into a single array self.data"
        data = []
        max_length = 100
        for file in files:
            one_day = self.load_day(file, scale)
            data.append(one_day)
            max_length = min(max_length, len(one_day))
        self.data = np.stack([d[:max_length] for d in data], axis=0).squeeze()


def download_dataset(at_name, project_name):
    "Downloads dataset from wandb artifact"

    def _get_dataset(run):
        artifact = run.use_artifact(at_name, type="dataset")
        return artifact.download()

    if wandb.run is not None:
        run = wandb.run
        artifact_dir = _get_dataset(run)
    else:
        run = wandb.init(project=project_name, job_type="download_dataset")
        artifact_dir = _get_dataset(run)
        run.finish()

    files = ls(Path(artifact_dir))
    return files


if __name__ == "__main__":
    files = download_dataset(DATASET_ARTIFACT, project_name=PROJECT_NAME)
    train_ds = CloudDataset(files)
    print(f"Let's grab 5 samples: {train_ds[0:5].shape}")
