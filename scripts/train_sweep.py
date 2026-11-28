from scripts.script_utils import run_sweep

if __name__ == "__main__":
    server = "mine"
    max_jobs = 2
    script_methods = [
        # ("train.cl", "cl"),
        # *[("train.os", f"client{i}_os") for i in range(5)],
        ("train.fed_diffuse", "fed_diffuse"),
        # ("train.vq", "codebooks"),
        # ("train.pfdm", "pfdm"),
        # ("train.vq", "vq"),
    ]

    datasets = [
        # "cifar10",
        # "fmnist",
        "celeba",
        # "tiny_imagenet",
    ]

    grid = {
        "seed": [0],
        "dataset": datasets,
    }

    run_sweep(script_methods, server, max_jobs, grid)
