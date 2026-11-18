import logging

from cleanfid import fid

local = True
local = False

def compute_fid(img_path):
    score = fid.compute_fid(
        img_path,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        num_workers=0,
    )

    print(f"FID: {score:.4f}")

if __name__ == "__main__":
    img_path = "results/imgs/all"
    if local:
        for client in range(10):
            logging.info(f"Client {client}")
            compute_fid(f'{img_path}/client{client}')
    else:
        compute_fid(img_path)
    # for client in range(5):
        # compute_local_fid(client)
