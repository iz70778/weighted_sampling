import os
import torch
from config import args, global_model, client_loaders, linear_scheduler, logger
from torchvision.utils import save_image

if __name__ == "__main__":
    img_path = f"{args.results_path}/imgs/ws"

    model = global_model
    model.eval()

    client_num_samples = [
        len(client_loaders[client]) for client in range(args.num_clients)
    ]
    weights = [count / sum(client_num_samples) for count in client_num_samples]

    for client in range(args.num_clients):
        os.makedirs(f"{img_path}/{client}", exist_ok=True)
        model.load_state_dict(
            torch.load(f"{args.results_path}/models/os/client{client}.pth", )
        )
        num_samples = int(weights[client] * args.num_samplings)
        logger.info(f"Generating {num_samples} images for client {client}...")
        with torch.no_grad():
            for i in range(0, num_samples, args.test_batchsize):

                n = min(args.test_batchsize, num_samples - i)

                x = torch.randn(n, 3, args.image_size, args.image_size, device=args.device)

                for t in linear_scheduler.timesteps:
                    noise_pred = model(x, t).sample
                    x = linear_scheduler.step(noise_pred, t, x).prev_sample

                x = (x.clamp(-1, 1) + 1) / 2

                for j, img in enumerate(x):
                    save_image(img, f"{img_path}/{client}/{i + j:05d}.png")
