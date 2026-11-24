import os

from config import args
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import load_trained_model, sample_data

if __name__ == "__main__":
    img_path = f"{args.img_path}/avg_sample"
    os.makedirs(img_path, exist_ok=True)

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    # weights = get_weights(clients, client_loaders)
    weights = [1 / args.num_clients for _ in range(args.num_clients)]

    for client in clients:
        model = load_trained_model(f"client{client}_os")
        num_samplings = int(weights[client] * args.num_samplings)
        sample_data(model, f"{img_path}/{client}", num_samplings, f"Client {client}")
