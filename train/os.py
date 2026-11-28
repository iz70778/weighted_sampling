import torch

from common_utils import logger, get_diff_model, get_train_scheduler
from config import args
from data.data_loder import get_fl_train_dataloader, get_cl_train_dataloader, get_syn_train_dataloader
from train.train_utils import save_ema_model, get_checkpoint, get_ema_model, get_optimizer, get_warmup_steps

if __name__ == '__main__':
    method = args.method
    if method == "cl":
        train_loader = get_cl_train_dataloader()
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
    ema_model = get_ema_model(model)
    warmup_steps = get_warmup_steps(train_loader)
    train_scheduler = get_train_scheduler()

    check_path = args.check_path / method
    check_path.mkdir(parents=True, exist_ok=True)
    logger.add_log_path(check_path)
    checkpoint = get_checkpoint(check_path)

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

    model.train()
    for epoch in range(start_epoch + 1, args.client_epochs + 1):
        total_loss = 0.0
        for x0, _, _, _ in train_loader:
            x0 = x0.to(args.device)
            B = x0.shape[0]
            t = torch.randint(0, args.T, (B,), device=args.device).long()
            noise = torch.randn_like(x0)
            xt = train_scheduler.add_noise(x0, noise, t)
            noise_pred = model(xt, t).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            update_step += 1
            lr = args.lr * min(update_step / warmup_steps, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            ema_model.step(model.parameters())
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Method {method} - Epoch {epoch}: Loss = {avg_loss:.4f}")
        if epoch % 100 == 0 or epoch == args.client_epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update_step,
            }, f"{check_path}/{method}_{epoch}.pth")

    save_ema_model(model, ema_model, method)
