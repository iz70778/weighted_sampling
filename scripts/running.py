import multiprocessing as mp
import os
import subprocess

os.environ["PYTHONIOENCODING"] = "utf-8"

from tqdm import tqdm


def run_cmd(cmd, print_cmd):
    if print_cmd:
        print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        env=env
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_cmd(cmd):
    print("Running:", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        env=env
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def _init_worker():
    os.environ["PYTHONPATH"] = os.getcwd()


def run_all_tasks(cmds, num_workers=3):
    with mp.Pool(processes=num_workers, maxtasksperchild=1, initializer=_init_worker) as pool:
        try:
            list(tqdm(
                pool.imap(run_cmd, cmds),
                total=len(cmds),
                desc="Processing",
                unit="task"
            ))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            exit(1)

    print("All tasks completed.")
