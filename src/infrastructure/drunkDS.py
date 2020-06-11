import torch
import numpy as np
from glob import glob

from torch.utils.data import Dataset

class DrunkDS(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = np.array(glob(root+"/*"))

    def __getitem__(self, index):
        arr = torch.tensor((np.load(self.image_paths[index])))
        if "drunk" in self.image_paths[index]:
            return (arr, 1)
        else:
            return (arr, 0)
    
    def __len__(self):
        return len(self.image_paths)
