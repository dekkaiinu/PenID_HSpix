from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hspix_dataset import HsPixelDataset

from metrics import Accuracy
from model_wrapper import ModelWrapper
from logger import Logger
from models.mlp_batch_norm import MLP_BatchNotm


@hydra.main(version_base=None, config_path='cfg', config_name='config')
def main(cfg: DictConfig):
    training_dataset = HsPixelDataset(feature_path='../dataset/train_feature.npy', target_path='../dataset/train_target.npy')
    training_dataset = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=20, pin_memory=True, prefetch_factor=10)

    test_dataset = HsPixelDataset(feature_path='../dataset/validation_feature.npy', target_path='../dataset/validation_target.npy')
    test_dataset = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=20, pin_memory=True, prefetch_factor=10)

    # Init model
    model = MLP_BatchNotm(input_dim=cfg.input_dim, output_dim=cfg.output_dim, dropout_prob=cfg.dropout_prob)
    # Print number of parameters
    print("# parameters", sum([p.numel() for p in model.parameters()]))
    # Model to device
    model.to(cfg.device)
    # Init data parallel
    if cfg.data_parallel:
        model = nn.DataParallel(model)
    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_parameter.lr)
    # Init learning rate schedule
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train_parameter.epochs//10, gamma=cfg.train_parameter.scheduler_gamma)
    # Init loss function
    loss_function = nn.CrossEntropyLoss()
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 loss_function_test=nn.CrossEntropyLoss(),
                                 training_dataset=training_dataset,
                                 test_dataset=test_dataset,
                                 lr_schedule=lr_schedule,
                                 validation_metric=Accuracy(),
                                 logger=Logger(),
                                 device=cfg.device)
    # Perform training
    model_wrapper.train(epochs=cfg.train_parameter.epochs)


if __name__ == '__main__':
    main()