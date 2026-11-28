import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_task(script, server, params=None):
    server_configs = {
        "mine": (
            "F:/pycodes/weighted_sampling/results",
            "F:/data",
            "",
        ),
        "calder": (
            "/scratch-calder/users/sc21s2j/weighted_sampling/results",
            "/scratch-calder/users/sc21s2j/dataset",
            "module load calder/hopper miniforge3",
        ),
        "aire": (
            "/mnt/scratch/sc21s2j/weighted_sampling/results",
            "/mnt/scratch/sc21s2j/dataset",
            "module load miniforge",
        ),
        "swjtu_yll": (
            "/opt/data/private/ws/results",
            "/opt/data/private",
            "source /root/miniconda3/etc/profile.d/conda.sh\n"
            "conda activate ttt",
        ),
    }

    if server not in server_configs:
        print(f"Unknown server: {server}", file=sys.stderr)
        raise SystemExit(1)

    results_path, dataset_path, setup = server_configs[server]

    task_params = {
        "results_path": results_path,
        "dataset_path": dataset_path,
        **(params or {}),
    }

    python_args = ["-u", "-m", script]
    for key, value in task_params.items():
        python_args.extend([f"--{key}", str(value)])

    if setup:
        cmd = [
            "bash", "-c",
            f'set -eo pipefail\n{setup}\nexec python "$@"',
            "run_task",
            *python_args,
        ]
    else:
        cmd = [sys.executable, *python_args]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{pythonpath}"
        if pythonpath else str(PROJECT_ROOT)
    )

    dataset = task_params.get("dataset") or "cifar10"
    method = task_params.get("method") or script.rsplit(".", 1)[-1]

    done_file = PROJECT_ROOT / task_params["results_path"] / dataset / "done.txt"
    print(done_file)
    task_key = f"{script} {method}"
    try:
        done = set(done_file.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        done = set()
    if task_key in done:
        print(f"Already done for {cmd[3:]!r}")
        return 0

    print(f"Start for {cmd[3:]!r}")

    with subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()

    if return_code == 0 and done_file is not None:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with done_file.open("a", encoding="utf-8") as file:
            file.write(f"{task_key}\n")

    return return_code


def run_sweep(script_methods, server, max_jobs, grid):
    combinations = []
    for entry in script_methods:
        script, method = entry
        task_grid = {**grid, "method": [method]}

        combinations.extend(
            (script, dict(zip(task_grid, values)))
            for values in itertools.product(*task_grid.values())
        )

    print(f"Total experiments: {len(combinations)}")
    for i, (script, params) in enumerate(combinations, start=1):
        print(f"[{i}/{len(combinations)}]")

    failed = 0

    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(run_task, script, server, params)
            for script, params in combinations
        ]

        for future in as_completed(futures):
            if future.result() != 0:
                failed += 1

    if failed:
        raise SystemExit(f"{failed} task(s) failed.")

    print("All tasks completed.")
