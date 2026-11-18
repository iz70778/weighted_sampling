from config import args
from train import client
from train.client import train_diff_os
from scripts.init import global_model, client_loaders

if __name__ == '__main__':
    client = args.client_id
    train_diff_os(client, client_loaders[client], global_model)