####################
# Import Libraries
####################
import os
import sys
from PIL import Image
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import model_selection
from sklearn import metrics
import albumentations as A
import timm
from omegaconf import OmegaConf
import wandb

####################
# Utils
####################
def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

####################
# Config
####################
conf_dict = {'batch_size': 8,#32, 
             'epoch': 30,
             'height': 224,#640,
             'width': 224,
             'model_name': 'efficientnet_b0',
             'lr': 1e-5,
             'fold': 0,
             'drop_rate': 0.2,
             'drop_path_rate': 0.2,
             'data_dir': '../input/petfinder-pawpularity-score/',
             'model_path': None,
             'output_dir': './',
             'pseudo': None,
             'seed': 2021,
             'group': None,
             'trainer': {}}
conf_base = OmegaConf.create(conf_dict)

####################
# Dataset
####################
class PawpularDataset(Dataset):
    def __init__(self, df, transform=None, conf=None):
        
        self.df = df.reset_index(drop=True)
        self.dir_names = df['dir'].values
        self.targets = df['Pawpularity'].values
        self.transform = transform
        self.conf = conf
        self.dense_features = df[['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
                               'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'Id']
        image = cv2.imread(os.path.join(self.dir_names[idx],"{}.jpg".format(img_id)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[idx, :]
        targets = self.targets[idx]
        
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor([targets], dtype=torch.float)
        }
    
####################
# Data Module
####################

class SETIDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf  

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(os.path.join(self.conf.data_dir, "train.csv"))
            df['dir'] = os.path.join(self.conf.data_dir, "train")
            
            # cv split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.conf.seed)
            for n, (train_index, val_index) in enumerate(skf.split(df, df["Pawpularity"])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)
            
            train_df = df[df['fold'] != self.conf.fold]
            valid_df = df[df['fold'] == self.conf.fold]

            #if self.conf.pseudo is not None:
            #    pseudo_df = pd.read_csv(self.conf.pseudo)
            #    pseudo_df['dir'] = os.path.join(self.conf.data_dir, "test")
            #    train_df = pd.concat([train_df, pseudo_df])
            '''
            train_transform = A.Compose([
                A.LongestMaxSize(max_size=self.conf.height),
                A.PadIfNeeded(min_height=self.conf.height, min_width=self.conf.width, value=0),
                #A.Resize(height=self.conf.height, width=self.conf.width, p=1),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0,),
                            ],p=1.0)
            '''
            train_transform = A.Compose([
                #A.LongestMaxSize(max_size=self.conf.height),
                #A.PadIfNeeded(min_height=self.conf.height, min_width=self.conf.width, value=0),
                #A.Resize(height=self.conf.height, width=self.conf.width, p=1),
                A.SmallestMaxSize(self.conf.height),
                A.CenterCrop(self.conf.height, self.conf.height),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.OneOf([
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.),
                        A.ElasticTransform(alpha=3),
                    ], p=0.20),
                A.OneOf([
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        A.MedianBlur(),
                    ], p=0.20),
                #A.OneOf([
                #        A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                #        A.Downscale(scale_min=0.75, scale_max=0.95),
                #    ], p=0.2),
                A.IAAPiecewiseAffine(p=0.2),
                A.Cutout(max_h_size=int(self.conf.height * 0.1), max_w_size=int(self.conf.width * 0.1), num_holes=5, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0,),
                            ],p=1.0)

            valid_transform = A.Compose([
                #A.LongestMaxSize(max_size=self.conf.height),
                #A.PadIfNeeded(min_height=self.conf.height, min_width=self.conf.width, value=0),
                #A.Resize(height=self.conf.height, width=self.conf.width, p=1),
                A.SmallestMaxSize(self.conf.height),
                A.CenterCrop(self.conf.height, self.conf.height),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0,),
                            ],p=1.0)

            self.train_dataset = PawpularDataset(train_df, transform=train_transform,conf=self.conf)
            self.valid_dataset = PawpularDataset(valid_df, transform=valid_transform, conf=self.conf)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
    
####################
# Lightning Module
####################

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=1, pretrained=True, in_chans=3)
        num_features = self.model.num_features
        self.model.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 1))
        if self.hparams.model_path is not None:
            print(f'load model path: {self.hparams.model_path}')
            self.model = load_pytorch_model(self.hparams.model_path, self.model, ignore_suffix='model')
        #self.criteria = torch.nn.MSELoss()
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        #optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=self.hparams.lr)
        #optimizer = torch.optim.Adam([{'params': nn.Sequential(*list(self.model.modules())[:-2]).parameters(), 'lr': self.hparams.lr*0.1},
        #                              {'params': self.model.classifier.parameters(), 'lr': self.hparams.lr}], lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.epoch, eta_min=1e-4)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, y = batch['images'], batch['features'], batch['targets']
        y = y/100.0

        if torch.rand(1)[0] < 0.5: #self.current_epoch < self.hparams.epoch*1.8:
            # mixup
            alpha = 0.5
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size)
            x = lam * x + (1 - lam) * x[index, :]
            #y = lam * y +  (1 - lam) * y[index]

            y_hat = self.model(x)
            loss = lam * self.criteria(y_hat, y) + (1 - lam) * self.criteria(y_hat, y[index])
        else:
            y_hat = self.model(x)
            #loss = self.criteria(y_hat, y.view(-1, 1))
            loss = self.criteria(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, f, y = batch['images'], batch['features'], batch['targets']
        y_hat = self.model(x)
        #loss = self.criteria(y_hat, y.view(-1, 1))
        loss = self.criteria(y_hat, y/100.0)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu().detach().numpy()
        y_hat = torch.cat([x["y_hat"].sigmoid()*100 for x in outputs]).cpu().detach().numpy()

        val_score = metrics.mean_squared_error(y, y_hat, squared=False)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_score', val_score)
        
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(conf.seed)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, conf.group, 'fold' + str(conf.fold), 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, conf.group, 'fold' + str(conf.fold), 'csv_log/'))
    wandb_logger = loggers.WandbLogger(project='PetFinder2021', log_model=False, group=conf.group)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, conf.group, 'fold' + str(conf.fold), 'ckpt/'), monitor='val_score', 
                                          save_last=True, save_top_k=1, mode='min', 
                                          save_weights_only=True, filename=f'fold{conf.fold}-'+'{epoch}-{val_score:.5f}')

    data_module = SETIDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger, wandb_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()
