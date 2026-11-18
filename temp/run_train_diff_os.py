from scripts.running import run_all_tasks


if __name__ == "__main__":
    tasks = []

    seed = 0
    dataset = 'cifar10'
    dirichlet_alpha = 0.1
    num_clients = 1

    for client_id in range(num_clients):
        cmd = [
            "python", "-u", "scripts/train_diff_os.py",
            "--seed", str(seed),
            "--dataset", dataset,
            "--dirichlet_alpha", str(dirichlet_alpha),
            "--num_clients", str(num_clients),
            "--client_id", str(client_id),
            "--server", "mine"
        ]
        tasks.append(cmd)


    run_all_tasks(tasks, num_workers=1)
