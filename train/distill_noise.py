import torch
import torch.nn.functional as F

from common_utils import get_diff_model, get_train_scheduler, get_weights, logger
from config import args
from data.data_loder import get_fl_train_dataloader
from synthesis.syn_utils import load_trained_model
from train.train_utils import get_checkpoint, get_ema_model, get_optimizer, get_warmup_steps, save_ema_model

if __name__ == "__main__":
    mode = "direct_ddim"  # direct_ddim | direct_ddpm | pool
    is_init_client = True

    device = torch.device(args.device)
    img_shape = (3, args.image_size, args.image_size)
    bs = args.batchsize
    steps = int(getattr(args, "distill_steps_per_epoch", 100))
    ddim_steps = int(getattr(args, "distill_ddim_steps", 50))
    pool_size = int(getattr(args, "distill_pool_size", 2 * bs))
    last = args.T - 1

    scheduler = get_train_scheduler()
    grid = range(last, -1, -1)
    if mode == "direct_ddim":
        alpha = scheduler.alphas_cumprod.to(device)
        if last > 0:
            jumps = min(ddim_steps, last)
            grid = [last - i * last // jumps for i in range(jumps + 1)]


    @torch.no_grad()
    def sample_to(teacher, targets):
        """Start at T-1. Stop each sample exactly at its requested training timestep."""
        x = torch.randn((len(targets), *img_shape), device=device)
        smallest = int(targets.min())
        for now, following in zip(grid, grid[1:]):
            if now <= smallest:
                break
            rows = torch.where(targets < now)[0]
            active = x[rows]
            eps = teacher(active, now).sample
            if mode == "direct_ddim":
                dest = targets[rows].clamp(min=following).to(device)
                a = alpha[now]
                b = alpha[dest].view(-1, 1, 1, 1)
                x0 = (active - (1 - a).sqrt() * eps) / a.sqrt()
                x[rows] = b.sqrt() * x0 + (1 - b).sqrt() * eps
            else:
                x[rows] = scheduler.step(eps, now, active).prev_sample
        return x


    clients = list(range(args.num_clients))
    teachers = [load_trained_model(f"client{i}_os").to(device).eval().requires_grad_(False)
                for i in clients]
    client_loaders = get_fl_train_dataloader()
    weights = torch.as_tensor(get_weights(clients, client_loaders), dtype=torch.float32)
    del client_loaders

    method = f"distill_{mode}_initclient_{int(is_init_client)}"
    if mode == "pool":
        method += f"_poolsize_{pool_size}"
    elif mode == "direct_ddim":
        method += f"_jumps_{ddim_steps}"
    method += f"_c{args.num_clients}_a{args.dirichlet_alpha}_seed{args.seed}"
    check_path = args.check_path / method
    check_path.mkdir(parents=True, exist_ok=True)
    logger.add_log_path(check_path)
    checkpoint = get_checkpoint(check_path)

    student = get_diff_model().to(device)
    if checkpoint is None and is_init_client:
        # init_client = random.choice(clients)
        init_client = 0
        student.load_state_dict(teachers[init_client].state_dict())
        logger.info(f"Initializing student from client {init_client}")
    optimizer = get_optimizer(student)
    ema = get_ema_model(student)
    epoch_done, update_step = 0, 0
    pools = {}

    if checkpoint is not None:
        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        epoch_done, update_step = checkpoint["epoch"], checkpoint["update_step"]
        pools = {i: (x.to(device), t.cpu()) for i, (x, t) in checkpoint["pools"].items()}
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
        del checkpoint
        logger.info(f"Resuming after epoch {epoch_done} (update {update_step})")
    elif mode == "pool":
        for client, teacher in enumerate(teachers):
            if weights[client] == 0:
                continue
            pool_t = torch.randint(args.T, (pool_size,), device="cpu")
            pool_x = torch.empty((pool_size, *img_shape), device=device)
            for start in range(0, pool_size, bs):
                chunk = slice(start, start + bs)
                pool_x[chunk] = sample_to(teacher, pool_t[chunk])
            pools[client] = pool_x, pool_t

    warmup = max(1, get_warmup_steps(range(steps)))
    student.train()
    for epoch in range(epoch_done + 1, args.client_epochs + 1):
        total_loss = torch.zeros((), device=device)
        for _ in range(steps):
            with torch.no_grad():
                owners = torch.multinomial(weights, bs, replacement=True)
                batch_x = torch.empty((bs, *img_shape), device=device)
                batch_t = torch.empty(bs, dtype=torch.long, device=device)
                teacher_eps = torch.empty_like(batch_x)
                for client, teacher in enumerate(teachers):
                    rows = torch.where(owners == client)[0]
                    if rows.numel() == 0:
                        continue
                    if mode == "pool":
                        pool_x, pool_t = pools[client]
                        slots = torch.randperm(pool_size)[:len(rows)]
                        xt, times = pool_x[slots], pool_t[slots]
                    else:
                        times = torch.randint(args.T, (len(rows),), device="cpu")
                        xt = sample_to(teacher, times)

                    model_times = times.to(device)
                    target = teacher(xt, model_times).sample
                    batch_x[rows], batch_t[rows], teacher_eps[rows] = xt, model_times, target

                    if mode == "pool":
                        # local indexes xt/times/target; slots[local] indexes the client's pool.
                        for now in times.unique().tolist():
                            local = torch.where(times == now)[0]
                            pos = slots[local]
                            if now == 0:
                                pool_x[pos] = torch.randn_like(xt[local])
                                pool_t[pos] = last
                            else:
                                pool_x[pos] = scheduler.step(target[local], now, xt[local]).prev_sample
                                pool_t[pos] = now - 1

            optimizer.zero_grad(set_to_none=True)
            student_eps = student(batch_x, batch_t).sample
            loss = F.mse_loss(student_eps, teacher_eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            update_step += 1
            for group in optimizer.param_groups:
                group["lr"] = args.lr * min(update_step / warmup, 1.0)
            optimizer.step()
            ema.step(student.parameters())
            total_loss += loss.detach()

        logger.info(f"{method} - Epoch {epoch}: Loss = {total_loss.item() / steps:.6f}")
        if epoch % 10 == 0 or epoch == args.client_epochs:
            dest = check_path / f"{method}_{epoch}.pth"
            tmp = dest.with_suffix(".tmp")
            torch.save({
                "epoch": epoch, "update_step": update_step,
                "model": student.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "pools": pools,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }, tmp)
            tmp.replace(dest)

    ema.copy_to(student.parameters())
    torch.save(
        student.state_dict(),
        args.model_path / f"{method}.pth",
    )
