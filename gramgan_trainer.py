import os
import sys
import shlex
import subprocess
from datetime import datetime
from typing import List

# --------------------------------------------------------------------
# Configurations (edit as needed)
# --------------------------------------------------------------------
default_configure = {
    "hr_dir": "./data/DIV2K_train_HR",
    "batch_size": 16,
    "base_lr": 2e-4,
    "max_patch_db": 100000,
    "total_iters": 200000,
    "output_dir": "./output_gram_gan",
    "save_path": "./output_gram_gan/gramgan_cpu_test.pth"
}

test_configure = {
    "hr_dir": r"D:\Projects\PYTHONPROJECTS\downloaded_images",
    "batch_size": 8,
    "base_lr": 1e-4,
    "max_patch_db": 1000,
    "total_iters": 2500,
    "output_dir": r"D:\Projects\PYTHONPROJECTS\output_test_run",
    "save_path": r"D:\Projects\PYTHONPROJECTS\output_test_run\gramgan_cpu_test.pth"
}

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def stream_subprocess(cmd_list: List[str], log_path: str, env=None) -> int:
    """
    Run subprocess with streaming stdout/stderr -> console and log file.
    Returns the process return code.
    """
    print(f"[Run] {' '.join(shlex.quote(x) for x in cmd_list)}")
    ensure_dir(os.path.dirname(log_path) or ".")
    with open(log_path, "w", encoding="utf-8", errors="ignore") as logf:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(line, end="")
                    logf.write(line)
                proc.wait()
            else:
                proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        return proc.returncode

def tail_file(path: str, n:int=200):
    """Return last n lines from file (safe)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception as e:
        return f"<unable to read log file: {e}>"

# --------------------------------------------------------------------
# Launcher class
# --------------------------------------------------------------------
class TrainGramGAN:
    def __init__(self, conf=None):
        self.conf = conf or default_configure
        self.hr_dir = self.conf["hr_dir"]
        self.batch_size = self.conf["batch_size"]
        self.base_lr = self.conf["base_lr"]
        self.max_patch_db = self.conf["max_patch_db"]
        self.total_iters = self.conf["total_iters"]
        self.output_dir = self.conf["output_dir"]
        self.save_path = self.conf.get("save_path", os.path.join(self.output_dir, "gramgan_cpu_test.pth"))

        # Prepare directories
        ensure_dir(self.output_dir)
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        ensure_dir(self.ckpt_dir)

        print(f"[Init] output_dir: {self.output_dir}")
        print(f"[Init] checkpoint dir: {self.ckpt_dir}")
        print(f"[Init] save_path: {self.save_path}")

    def train(self):
        print("[Train] Starting Gram-GAN training...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"train_{timestamp}.log")

        # Build command using current Python interpreter, target gramgan.py with args
        cmd_list = [
            sys.executable,
            "gramgan.py",
            "--hr_dir", self.hr_dir,
            "--batch_size", str(self.batch_size),
            "--total_iters", str(self.total_iters),
            "--save_path", self.save_path
        ]

        # Run and stream output
        returncode = stream_subprocess(cmd_list, log_file)

        if returncode == 0:
            print(f"[Train] Completed successfully. Checkpoint saved at: {self.save_path}")
            print(f"[Train] Log: {log_file}")
        else:
            print(f"[Error] Training script exited with code {returncode}")
            print(f"[Error] Showing last 200 lines of log ({log_file}):\n")
            print(tail_file(log_file, n=200))
            raise RuntimeError(f"Training script failed (exit code {returncode}). See log: {log_file}")

# --------------------------------------------------------------------
# Run example (use test_configure for a quick run)
# --------------------------------------------------------------------
if __name__ == "__main__":
    trainer = TrainGramGAN(conf=test_configure)
    trainer.train()