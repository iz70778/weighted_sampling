from config import args
from data.data_loder import get_fl_train_dataloader, get_cl_train_dataloader, get_syn_train_dataloader
from train.train_utils import train_diff_model

if __name__ == '__main__':
    method = args.method
    if method == "cl":
        train_dataloader = get_cl_train_dataloader()
    elif method == "cl_distill":
        train_dataloader = get_syn_train_dataloader()
    if method.endswith("_os"):
        client = method.removeprefix("client").removesuffix("_os")
        client_loaders = get_fl_train_dataloader()
        train_loader = client_loaders[client]
    else:
        raise ValueError(f"Unknown method: {args.method}")

    train_diff_model(train_loader, method)
