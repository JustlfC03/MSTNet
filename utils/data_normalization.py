import os
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)
import nibabel as nib
import numpy as np
import torch
from os.path import join

import os


def adaptive_normal(img):
    min_p = 0.001
    max_p = 0.999  # quantile prefer 98~99

    imgArray = img
    imgPixel = imgArray[imgArray >= 0]
    imgPixel, _ = torch.sort(imgPixel)
    index = int(round(len(imgPixel) - 1) * min_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_min = imgPixel[index]

    index = int(round(len(imgPixel) - 1) * max_p + 0.5)
    if index < 0:
        index = 0
    if index > (len(imgPixel) - 1):
        index = len(imgPixel) - 1
    value_max = imgPixel[index]

    mean = (value_max + value_min) / 2.0
    stddev = (value_max - value_min) / 2.0

    imgArray = (imgArray - mean) / stddev
    imgArray[imgArray < -1] = -1.0
    imgArray[imgArray > 1] = 1.0

    return imgArray
