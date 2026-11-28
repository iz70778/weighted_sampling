import os

from config import args
from synthesis.syn_utils import load_trained_model, sample_data

if __name__ == "__main__":
    img_path = args.img_path / args.method
    os.makedirs(img_path, exist_ok=True)
    model = load_trained_model(args.method)
    sample_data(model, img_path, args.num_samplings)

