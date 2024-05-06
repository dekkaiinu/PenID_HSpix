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
    def __init__(self, root: Union[str, Path], meta_data: bool = False, random_seed: int = 0, aug_beta = False):
        super().__init__()
        self.dataset = self._load_data(root)

        self.data = np.array([single_data['data'] for single_data in self.dataset])
        self.target = np.array([single_data['target'] for single_data in self.dataset])

        self.label_encoder = LabelEncoder()
        if len(self.target.shape) > 1:
            self.target = self.target[:, 0]
        self.target = self.label_encoder.fit_transform(self.target)

        self.meta_data = meta_data
        if self.meta_data:
            self.meta_data = np.array([single_data['meta_data'] for single_data in self.dataset])
        if aug_beta:
            self.beta = aug_beta

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        '''
        '''
        hs_pixel, target = self.data[index, :], np.array(self.target[index])

        hs_pixel = torch.from_numpy(hs_pixel).type(torch.float32)
        target = torch.from_numpy(target).type(torch.LongTensor)

        if self.meta_data:
            meta_data = self.meta_data[index]
            return hs_pixel, target, meta_data

        return hs_pixel, target

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_data(self, root):
        with open(root, 'r') as file:
            data = json.load(file)
        return data
        
    def collate_fn(self, batch):
        batch_data, batch_target, batch_meta_data = zip(*batch)
        batch_data = torch.stack([self.relight_aug(feature=hs_pixel, target = target, meta_data=meta_data, beta=self.beta) 
                                  for hs_pixel, target, meta_data in zip(batch_data, batch_target, batch_meta_data)], dim=0)
        return batch_data, torch.tensor(batch_target)
    
    def relight_aug(self, feature, target, meta_data, beta):
        target_indices = [index for index, item in enumerate(self.dataset) if (item['meta_data'] == meta_data) and (item['target'] != target)]
        selected_indices = random.choice(target_indices)
        pair_data = self.dataset[selected_indices]
        pair_pix = pair_data['data']
        pair_id = pair_data['target']

        target_indices = [index for index, item in enumerate(self.dataset) if (item['meta_data'] != meta_data) and (item['target'] == pair_id)]
        selected_indices = random.choice(target_indices)
        differ_data = self.dataset[selected_indices]
        differ_pix = differ_data['data']

        if differ_pix and pair_pix:

            light_rate = np.array(differ_pix) / np.array(pair_pix)
            spector_relight = feature * light_rate

            transformed_spector = beta * feature + (1 - beta) * spector_relight
            return transformed_spector
        else:
            return feature