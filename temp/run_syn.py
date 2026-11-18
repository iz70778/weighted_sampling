from scripts.running import run_all_tasks, run_cmd


if __name__ == "__main__":
    tasks = []

    seed = 0
    dataset = 'cifar10'
    dirichlet_alpha = 0.1
    num_clients = 5

    for client_id in range(num_clients):
        cmd = [
            "python", "-u", "-m", "synthesis.dfd",
            "--seed", str(seed),
            "--dataset", dataset,
            "--dirichlet_alpha", str(dirichlet_alpha),
            "--num_clients", str(num_clients),
            "--client_id", str(client_id),
            "--server", "mine"
        ]
        tasks.append(cmd)

    # run_cmd(tasks[0])
    run_all_tasks(tasks, num_workers=1)
