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
    def __init__(self, feature_path: Union[str, Path], target_path: Union[str, Path]):
        super().__init__()
        self.feature = np.load(feature_path)
        print(self.feature.shape)
        self.target = np.load(target_path)
        print(self.target.shape)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        '''
        '''
        hs_pixel, target = self.feature[index, :], np.array(self.target[index])

        hs_pixel = torch.from_numpy(hs_pixel / 4095).type(torch.FloatTensor)
        target = torch.from_numpy(target).type(torch.LongTensor)

        return hs_pixel, target

    def __len__(self) -> int:
        return self.target.shape[0]