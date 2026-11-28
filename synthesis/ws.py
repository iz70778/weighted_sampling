import os

from common_utils import get_weights, logger
from config import args
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import load_trained_model, sample_data

if __name__ == "__main__":

    img_path = args.img_path / args.method
    os.makedirs(img_path, exist_ok=True)

    clients = list(range(args.num_clients))
    client_loaders = get_fl_train_dataloader()

    if args.method == "ws":
        weights = get_weights(clients, client_loaders)
    else:
        weights = [1 / args.num_clients for _ in range(args.num_clients)]

    for client in clients:
        model = load_trained_model(f"client{client}_os")
        num_samplings = int(weights[client] * args.num_samplings)
        logger.info(f"Generating {num_samplings} images for Client {client}...")
        sample_data(model, img_path / str(client), num_samplings)
