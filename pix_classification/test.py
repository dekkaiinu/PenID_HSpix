from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hspix_dataset import HsPixelDataset
from models.mlp_batch_norm import MLP_BatchNotm

from utils.output_metric import *
from utils.AvgrageMeter import *
from utils.accuracy import *

@hydra.main(version_base=None, config_path='cfg', config_name='config')
def test(cfg: DictConfig):
    test_dataset = HsPixelDataset(feature_path='../dataset/test_feature_rgb.npy', target_path='../dataset/test_target_rgb.npy')
    test_dataset = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)


    model = MLP_BatchNotm(input_dim=cfg.input_dim, output_dim=cfg.output_dim, dropout_prob=cfg.dropout_prob)
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()
    model.to(cfg.device)

    loss_function = nn.CrossEntropyLoss()

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_data, batch_target in test_dataset:
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data)
        
        loss = loss_function(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    OA, AA_mean, AA = output_metric(tar, pre)
    print('OA: {:.4f} | AA: {:.4f}'.format(OA, AA_mean))
    print('AA:', AA)

if __name__ == '__main__':
    test()