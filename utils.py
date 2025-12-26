import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.utils import make_grid
from PIL import Image
from itertools import islice
from torch import Tensor


DATASET = "deadprogram/clothes-with-class"
DATA_DIR = "data"


def get_batch(
        N: int, 
        size: tuple[int] = (96, 96),
        dataset_name: str = DATASET, 
        split: str = "train"
    ) -> Tensor:
    ## TODO: seperate dataset dependency
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )
    data = Dataset.from_list(list(islice(dataset, N)))
    transform = T.Compose([
        T.Resize(size=size),
        T.ToTensor(),
    ])
    return torch.concat([transform(img["image"]).unsqueeze(0) for img in data], dim=0)

