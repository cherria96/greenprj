import sys
sys.path.append('/Users/sujinchoi/Desktop/AD_TSformer/sujineeda')
from utils.dataloader import TimeSeriesDataset
from models.ns_Transformer import ns_TimeSeriesForecasting
from models.configs import Config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
import torch
torch.set_default_dtype(torch.float32)
import wandb
wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")
from pytorch_lightning.loggers import WandbLogger
import random
import os

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything(100)

if __name__ == "__main__":
    ########## parameters ############
    data_name = "AD"
    accelerator = "cpu"
    wandblogging = False
    if data_name == "AD": 
        data_type = "sbk_AD_B"
        size = {
            'seq_len': 30,
            'label_len': 15,
            'pred_len': 15
        }

        path = "data/sbk_ad_shift_concat.csv"
        enc_in = 19
        dec_in = 19
        c_out = 19
        freq = 'd'
        d_model = 32
        d_ff = 64

    elif data_name == 'ETTh1':
        data_type = "benchmark"
        size = {
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 48
        }

        path = "data/ETTh1.csv"
        enc_in = 7
        dec_in = 7
        c_out = 7
        freq = 'h'
        d_model = 64
        d_ff = 128


    elif data_name == 'ETTm2':
        data_type = "benchmark"
        size = {
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 48
        }

        path = "data/ETTm2.csv"
        enc_in = 7
        dec_in = 7
        c_out = 7
        freq = 't'
        d_model = 512
        d_ff = 2048
    elif data_name == 'ILL':
        data_type = "benchmark"
        size = {
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 48
        }

        path = "data/national_illness.csv"
        enc_in = 7
        dec_in = 7
        c_out = 7
        freq = 't'
        d_model = 512
        d_ff = 2048


    #####################################

    config = Config(
        data_name = data_name,
        path= path,
        seq_len= size['seq_len'],
        label_len= size['label_len'],
        pred_len= size['pred_len'],
        variate= 'ms',
        scale= True,
        is_timeencoded= True,
        random_state= 42,
        output_attention = False,
        enc_in = enc_in,
        d_model = d_model,
        embed = 'fixed',
        freq = freq,
        dropout = 0.05,
        dec_in = dec_in,
        factor = 1,
        n_heads = 8,
        d_ff = d_ff,
        activation = 'gelu',
        e_layers = 2,
        d_layers = 1,
        c_out = c_out,
        batch_size= 32,
        epoch= 20,
        lr= 0.0005,
        loss= 'mse',
        scheduler= 'exponential',
        inverse_scaling = False,
        num_workers = 0,
    )

    wandb_config = config.to_dict()

    train_data = TimeSeriesDataset(
                path = config.path,
                split="train",
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
            )

    val_data = TimeSeriesDataset(
                path = config.path,
                split="val",
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
            )

    test_data = TimeSeriesDataset(
                path = config.path,
                split="test",
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
            )
    
    train_dataloader = DataLoader(
                        train_data,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers,
                        )
    val_dataloader = DataLoader(
                        val_data,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.num_workers,
                        )
    test_dataloader = DataLoader(
                        test_data,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.num_workers,
                        )
    model = ns_TimeSeriesForecasting(config, scaler = train_data)
    model_name = "NSFormer"
    logger = None
    if wandblogging:
        wandb_logger = WandbLogger(project = f'{model_name}_{data_name}', name = f"{model_name}_{config.epoch}_{config.seq_len}_{config.label_len}_{config.pred_len}", config=wandb_config)
        logger = wandb_logger
    trainer = L.Trainer(max_epochs = config.epoch, logger = logger, accelerator=accelerator)
    trainer.fit(model = model, train_dataloaders= train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
    trainer.save_checkpoint(f"./weights/concat_{model_name}-{data_name}-{config.epoch}_{config.seq_len}_{config.label_len}_{config.pred_len}.ckpt")
    np.save(f'test_loss_scaled_{model_name}_{data_name}.npy', model.test_loss)

# model = ns_TimeSeriesForecasting.load_from_checkpoint("weights/NSFormer-AD-20_96_96_96.ckpt")
# model.eval()


# Get predictions
predictions, targets = model.predict(test_dataloader)

# Visualize predictions
import matplotlib.pyplot as plt

num_cols = predictions.shape[-1]
fig, axes = plt.subplots(num_cols, 1, figsize=(15, num_cols * 3))

for i in range(num_cols):
    ax = axes[i]
    ax.plot(targets[0,:,i].cpu().numpy(), label='True')
    ax.plot(predictions[0,:,i].cpu().numpy(), label='Predicted')
    ax.legend()
    ax.set_title(f'Sample {i}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')



plt.tight_layout()
plt.savefig(f'plot_{model_name}_{data_name}.png')
plt.show()






