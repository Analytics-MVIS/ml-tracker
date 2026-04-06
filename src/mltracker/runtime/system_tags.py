from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Callable


def _try_cmd(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        return output if output else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_system_tags(command_runner: Callable[[list[str]], str | None] | None = None) -> dict[str, str]:
    run_cmd = command_runner if command_runner else _try_cmd

    tags: dict[str, str] = {
        "system.python": platform.python_version(),
        "system.hostname": socket.gethostname(),
    }

    git_commit = run_cmd(["git", "rev-parse", "HEAD"])
    if git_commit:
        tags["git.commit"] = git_commit

    git_branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if git_branch:
        tags["git.branch"] = git_branch

    nvidia_name = run_cmd([
        "nvidia-smi",
        "--query-gpu=name",
        "--format=csv,noheader",
    ])
    if nvidia_name:
        first_gpu = nvidia_name.splitlines()[0].strip()
        if first_gpu:
            tags["system.gpu"] = first_gpu

    cuda_version = os.getenv("CUDA_VERSION") or run_cmd(["nvcc", "--version"])
    if cuda_version:
        tags["system.cuda"] = cuda_version

    return tags
