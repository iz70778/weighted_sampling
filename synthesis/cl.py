import os

from config import args
from synthesis.syn_utils import load_trained_model, sample_data

if __name__ == "__main__":
    method = args.method
    img_path = f"{args.img_path}/{method}"
    os.makedirs(img_path, exist_ok=True)
    model = load_trained_model(method)
    args.batch_size = args.cl_batch_size
    sample_data(model, f"{img_path}", args.num_samplings)

