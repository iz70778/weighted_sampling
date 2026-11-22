import torch

from config import args
from common_utils import get_diff_model
from data.data_loder import get_fl_train_dataloader
from train.train_utils import train_diff_model

if __name__ == '__main__':
    client = args.client_id
    client_loaders = get_fl_train_dataloader()
    client_loader = client_loaders[client]

    model_name = f"client{client}_os"

    train_diff_model(client_loader, model_name)
