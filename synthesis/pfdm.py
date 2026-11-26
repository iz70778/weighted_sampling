import os

from common_utils import get_weights, logger
from config import args
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import load_trained_model, sample_data

if __name__ == "__main__":
    img_path = f"{args.img_path}/pdfm"
    os.makedirs(img_path, exist_ok=True)

    clients = list(range(args.num_clients))

    global_model = load_trained_model(f"pdfm")

    for client in clients:
        client_model = load_trained_model(f"client{client}_os")
        # num_samplings = int(weights[client] * args.num_samplings)
        # logger.info(f"Generating {num_samplings} images for Client {client}...")
        # sample_data(client_model, f"{img_path}/{client}", num_samplings)
