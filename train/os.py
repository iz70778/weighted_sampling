import os

import torch

from common_utils import logger, get_diff_model, get_train_scheduler
from config import args
from data.data_loder import get_fl_train_dataloader, get_cl_train_dataloader, get_syn_train_dataloader
from train.train_utils import train_epoch, save_model, get_checkpoint, get_ema_model, get_optimizer

if __name__ == '__main__':
    method = args.method
    if method == "cl":
        train_loader = get_cl_train_dataloader()
        args.batchsize = args.cl_batchsize
    elif method == "cl_distill":
        train_loader = get_syn_train_dataloader()
    elif method.endswith("_os"):
        client = method.removeprefix("client").removesuffix("_os")
        client = int(client)
        client_loaders = get_fl_train_dataloader()
        train_loader = client_loaders[client]
    else:
        raise ValueError(f"Unknown method: {args.method}")

    model = get_diff_model()

    optimizer = get_optimizer(model)

    check_path = f"{args.check_path}/{method}"
    os.makedirs(check_path, exist_ok=True)
    logger.add_log_path(check_path)

    checkpoint = get_checkpoint(check_path)

    ema_model = get_ema_model(model)

    train_scheduler = get_train_scheduler()

    if checkpoint is None:
        logger.info(f"Method {method} starts training from scratch")
        start_epoch = 0
        update_step = 0
    else:
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"]
        update_step = checkpoint["update_step"]

        if start_epoch >= args.client_epochs:
            logger.info(f"Method {method} has completed training")
        else:
            logger.info(f"Method {method} start training "
                        f"from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, args.client_epochs + 1):
        loss, update_step = train_epoch(model, optimizer, train_loader, train_scheduler, update_step)
        ema_model.step(model.parameters())
        logger.info(f"Method {method} - Epoch {epoch}: Loss = {loss:.4f}")
        if epoch % 100 == 0 or epoch == args.client_epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update_step,
            }, f"{check_path}/{method}_{epoch}.pth")

    save_model(model, ema_model, method)
