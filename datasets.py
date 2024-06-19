import os
import cv2
import torch
import random
from torch.utils import data


class VideoDatasets(data.Dataset):
    def __init__(self,  transforms=None):
        self.transform = transforms
        folder = "data/images/"
        files = os.listdir(folder)
        self.files = [folder+f for f in files]

    def __getitem__(self, index):
        frame_large = cv2.imread(self.files[index])  # large image
        h, w = frame_large.shape[0], frame_large.shape[1]
        offset_top = random.randint(0, h//6)
        offset_bottom = random.randint(0, h//6)
        offset_left = random.randint(0, w//6)
        offset_right = random.randint(0, w//6)
        frame_large = frame_large[offset_top:h-offset_top-offset_bottom, offset_left:w-offset_left-offset_right]

        if frame_large.shape[0] > 1024 or frame_large.shape[1] > 1024:
            max_size = frame_large.shape[0] if frame_large.shape[0] > frame_large.shape[1] else frame_large.shape[1]
            random_size = random.randint(600, 1280)
            ratio = random_size/max_size
            frame_large = cv2.resize(frame_large, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        offset_1 = -1*(frame_large.shape[0] % 4)
        offset_2 = -1*(frame_large.shape[1] % 4)
        if offset_1 != 0 and offset_2 != 0:
            frame_large = frame_large[:offset_1, :offset_2, :]
        elif offset_1 == 0 and offset_2 != 0:
            frame_large = frame_large[:, :offset_2, :]
        elif offset_1 != 0 and offset_2 == 0:
            frame_large = frame_large[:offset_1, :, :]

        width = frame_large.shape[1] // 4
        height = frame_large.shape[0] // 4
        frame_small = cv2.resize(frame_large, (width, height), interpolation=cv2.INTER_LINEAR)  # small image

        frame_large = frame_large.transpose((2, 1, 0))/255
        frame_small = frame_small.transpose((2, 1, 0))/255
        return torch.from_numpy(frame_large).float(), torch.from_numpy(frame_small).float()

    def __len__(self):
        return len(self.files)
