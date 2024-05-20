from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hspix_dataset import HsPixelDataset

from model_wrapper import ModelWrapper
from models.mlp_batch_norm import MLP_BatchNotm

from utils.create_runs_folder import *

@hydra.main(version_base=None, config_path='cfg', config_name='config')
def train(cfg: DictConfig):
    training_dataset = HsPixelDataset(feature_path='../dataset/train_feature_rgb.npy', target_path='../dataset/train_target_rgb.npy')
    training_dataset = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True)

    test_dataset = HsPixelDataset(feature_path='../dataset/validation_feature_rgb.npy', target_path='../dataset/validation_target_rgb.npy')
    test_dataset = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Init model
    model = MLP_BatchNotm(input_dim=cfg.input_dim, output_dim=cfg.output_dim, dropout_prob=cfg.dropout_prob)
    # Model to device
    model.to(cfg.device)
    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_parameter.lr)
    # Init learning rate schedule
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=cfg.train_parameter.epochs//10, 
                                                  gamma=cfg.train_parameter.scheduler_gamma)
    # Init loss function
    loss_function = nn.CrossEntropyLoss()
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 training_dataset=training_dataset,
                                 test_dataset=test_dataset,
                                 lr_schedule=lr_schedule,
                                 device=cfg.device)
    # Perform training
    save_path = create_runs_folder()
    model_wrapper.training_process(epochs=cfg.train_parameter.epochs, save_path=save_path)

if __name__=='__main__':
    train()