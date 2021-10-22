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

from tqdm import tqdm
import glob
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
    res = model.load_state_dict(new_state_dict, strict=False)
    print(res)
    return model

####################
# Config
####################
conf_dict = {'batch_size': 8,#32, 
             'height': 256,#640,
             'width': 256,
             'model_name': 'efficientnet_b0',
             'data_dir': '../input/petfinder-pawpularity-score/',
             'model_dir': None,
             'output_dir': './',
             'group': None,
             'seed': 2021}

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
            "targets": torch.tensor(targets, dtype=torch.float)
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
    def setup(self, stage=None, fold=0):
        if stage == 'fit':
            df = pd.read_csv(os.path.join(self.conf.data_dir, "train.csv"))
            df['dir'] = os.path.join(self.conf.data_dir, "train")
            
            # cv split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.conf.seed)
            for n, (train_index, val_index) in enumerate(skf.split(df, df["Pawpularity"])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)
            
            train_df = df[df['fold'] != fold]
            valid_df = df[df['fold'] == fold]

            #if self.conf.pseudo is not None:
            #    pseudo_df = pd.read_csv(self.conf.pseudo)
            #    pseudo_df['dir'] = os.path.join(self.conf.data_dir, "test")
            #    train_df = pd.concat([train_df, pseudo_df])
            
            train_transform = A.Compose([
                A.LongestMaxSize(max_size=self.conf.height),
                A.PadIfNeeded(min_height=self.conf.height, min_width=self.conf.width, value=0),
                #A.Resize(height=self.conf.height, width=self.conf.width, p=1),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0,),
                            ],p=1.0)

            valid_transform = A.Compose([
                #A.LongestMaxSize(max_size=self.conf.height),
                #A.PadIfNeeded(min_height=self.conf.height, min_width=self.conf.width, value=0),
                A.Resize(height=self.conf.height, width=self.conf.width, p=1),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],max_pixel_value=255.0,p=1.0,),
                            ],p=1.0)
            self.valid_df = valid_df
            self.train_dataset = PawpularDataset(train_df, transform=train_transform,conf=self.conf)
            self.valid_dataset = PawpularDataset(valid_df, transform=valid_transform, conf=self.conf)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
# ====================================================
# Inference function
# ====================================================
def inference(models, test_loader):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    with torch.no_grad():
        for i, (images) in tk0:
            images = images["images"].cuda()
            avg_preds = []
            for model in models:
                y_preds = model(images).sigmoid()*100

                avg_preds.append(y_preds.to('cpu').numpy())

            avg_preds = np.mean(avg_preds, axis=0)
            probs.append(avg_preds)
        probs = np.concatenate(probs)
    return probs
  
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    # get model path
    model_path = []
    for i in range(5):
        target_model = glob.glob(os.path.join(conf.model_dir, conf.group,  f'fold{i}/ckpt/*epoch*.ckpt'))
        #scores = [float(os.path.splitext(os.path.basename(i))[0].split('=')[-1]) for i in target_model]
        #model_path.append(os.path.join(conf.model_dir, conf.group,  f'fold{i}/ckpt/last.ckpt'))
        model_path += target_model
    #model_path = glob.glob(os.path.join(conf.model_dir, conf.group,  f'fold*/ckpt/last.ckpt'))
        
    models = []
    for ckpt in model_path:
      m = timm.create_model(model_name=conf.model_name, num_classes=1, pretrained=False, in_chans=3)
      num_features = m.num_features
      m.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 1))
      m = load_pytorch_model(ckpt, m, ignore_suffix='model')
      m.cuda()
      m.eval()
      models.append(m)

    
    # make oof
    oof_df = pd.DataFrame()
    for f, m in enumerate(models):
        data_module = SETIDataModule(conf)
        data_module.setup(stage='fit', fold=f)
        valid_df = data_module.valid_df
        valid_dataset = data_module.valid_dataset
        valid_loader =  DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
        predictions = inference([m], valid_loader)
        valid_df['preds'] = predictions
        oof_df = pd.concat([oof_df, valid_df])


    oof_score = metrics.mean_squared_error(oof_df['Pawpularity'], oof_df['preds'], squared=False)
    oof_df.to_csv(os.path.join(conf.output_dir, conf.group,  f"oof-{oof_score}.csv"), index=False)
        
    print(oof_score)
    print(oof_df.head())
    print(model_path)
    
    

if __name__ == "__main__":
    main()