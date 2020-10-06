import argparse
import collections
import torch
import numpy as np
from datasets.dataset import COCODataset
from models.vunet import VUnet
from trainer.trainer import Trainer
import yaml
import torch.optim as optim
from common import DEVICE
from torch.utils.data.dataloader import DataLoader
from datasets.MyDataset import MyDeepFashion

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config_file="./hyper-parameters.yaml"):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # setup data_loader instances
    train_dataset = COCODataset(index_file='/home/liuzhian/hdd/datasets/deepfashion/index.p',
                                spatial_size=config["model_pars"]["spatial_size"], phase='train')

    # train_dataset = MyDeepFashion('/home/liuzhian/hdd/datasets/deepfashion/index.p')

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False,
                                  num_workers=0, drop_last=True)
    valid_dataset = COCODataset(index_file='/home/liuzhian/hdd/datasets/deepfashion/index.p',
                                spatial_size=config["model_pars"]["spatial_size"], phase='valid')
    # valid_dataset = MyDeepFashion('/home/liuzhian/hdd/datasets/deepfashion/index.p', is_train=False)
    val_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=0, drop_last=False)

    # build model architecture, then print to console
    model = VUnet(config).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(0.5, 0.9))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    trainer = Trainer(config, model, optimizer,
                      train_dataloader,
                      val_dataloader,
                      log_dir="./log/dummy",
                      ckpt_path="./ckpt/dummy/vunet.pt",
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    main()
