import itertools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_task(script, server, params=None):
    bash = r"D:\Git\bin\bash.exe" if os.name == "nt" else "bash"

    cmd = [
        bash, "scripts/run.sh",
        "--script", script,
        "--server", server,
    ]

    if params:
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

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

        return process.wait()


def run_sweep(script, server, max_jobs,grid):
    # Cartesian product
    combinations = [
        dict(zip(grid.keys(), values))
        for values in itertools.product(*grid.values())
    ]

    print(f"Total experiments: {len(combinations)}")

    failed = 0

    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = [
            executor.submit(
                run_task, script,
                server, params,
            )
            for params in combinations
        ]

        for future in as_completed(futures):
            if future.result() != 0:
                failed += 1

    if failed:
        raise SystemExit(f"{failed} task(s) failed.")

    print("All tasks completed.")
