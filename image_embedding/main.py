import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from model import ShopeeNet
from data import prepare_loader
from config import GlobalConfig
from commons import AverageMeter
from scheduler import ShopeeScheduler

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_fold(df, config):
    df_folds = df.copy()
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed).split(
                        X=df_folds['image'], y=df_folds['label_group'])
    for fold, (train_idx, val_idx) in enumerate(skf):
        df_folds.loc[val_idx, 'fold'] = fold

    return df_folds



def train_one_epoch(train_loader, model, criterion, optimizer, device, scheduler, epoch, config):
    model.train()
    loss_score = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for step, (images, targets) in pbar:
        batch_size = images.shape[0]
        images = images.to(device)
        targets = targets.to(device)

        output = model(images,targets)
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_score.update(loss.detach().item(), batch_size)
        description = f"Epoch {epoch} Train Steps {step}/{len(train_loader)} train loss: {loss_score.avg:.3f}"
        pbar.set_description(description)

    if config.train_step_scheduler:
            scheduler.step()

    return loss_score


def validate_one_epoch(valid_loader, model, criterion, device, scheduler, epoch, config):
    model.eval()
    loss_score = AverageMeter()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    with torch.no_grad():
        for step, (images, targets) in pbar:
            batch_size = images.size()[0]
            images = images.to(device)
            targets = targets.to(device)

            output = model(images,targets)
            loss = criterion(output,targets)
            loss_score.update(loss.detach().item(), batch_size)
            description = f"Epoch {epoch} Valid Steps {step}/{len(valid_loader)} valid loss: {loss_score.avg:.3f}"
            pbar.set_description(description)

    return loss_score.avg




# MAIN
if __name__ == '__main__':
    config = GlobalConfig

    seed_everything(config.seed)
    train_csv = pd.read_csv(config.paths['csv_path'])

    df_folds = create_fold(train_csv, config)
    encoder = LabelEncoder()
    df_folds['label_group'] = encoder.fit_transform(df_folds['label_group'])


    for fold in range(config.num_folds):
        train_df = df_folds[df_folds['fold'] != fold].reset_index(drop=True)
        valid_df = df_folds[df_folds['fold'] == fold].reset_index(drop=True)
        train_loader, valid_loader = prepare_loader(train_df, valid_df, config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ShopeeNet(**config.model_params)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = ShopeeScheduler(optimizer,**config.scheduler_params)#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                  #T_0=config.num_epochs, T_mult=1, eta_min=1e-6, last_epoch=-1)

        best_loss = np.inf
        for epoch in range(config.num_epochs):
             train_loss = train_one_epoch(train_loader, model, criterion, optimizer, device, scheduler, epoch, config)
             valid_loss = validate_one_epoch(valid_loader, model, criterion, device, scheduler, epoch, config)

             if valid_loss < best_loss:
                 torch.save(model.state_dict(),f'model_{config.model_name}_IMG_SIZE_{config.img_size}\
                            _{config.model_params["loss_module"]}_fold{fold}.bin')
                 print(f'Loss improvement {best_loss} -> {valid_loss}')
                 best_loss = valid_loss
