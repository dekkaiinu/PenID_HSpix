import os.path
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder

class HsPixelDataset(data.Dataset):
    '''
    
    '''
    def __init__(self, feature, target):
        super().__init__()
        self.feature = feature
        self.target = target


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        '''
        '''
        hs_pixel, target = self.feature[index], self.target[index]

        hs_pixel = torch.tensor(hs_pixel, dtype=torch.float32) / 4095
        target = torch.tensor(target, dtype=torch.long)
    
        return hs_pixel, target

    def __len__(self) -> int:
        return len(self.target)