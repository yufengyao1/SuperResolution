import os
import cv2
import torch
from torch.utils import data


class VideoDatasets(data.Dataset):
    def __init__(self,  transforms=None):
        self.transform = transforms
        folder = "data/images/"
        files = os.listdir(folder)
        files.remove(".DS_Store")
        self.files = [folder+f for f in files]

    def __getitem__(self, index):
        frame_large = cv2.imread(self.files[index])  # large image
        width = int(frame_large.shape[1] / 4)
        height = int(frame_large.shape[0] / 4)
        frame_small = cv2.resize(frame_large, (width, height), interpolation=cv2.INTER_LINEAR)  # small image

        frame_large = frame_large.transpose((2, 1, 0))/255
        frame_small = frame_small.transpose((2, 1, 0))/255
        return torch.from_numpy(frame_large).float(), torch.from_numpy(frame_small).float()

    def __len__(self):
        return len(self.files)
