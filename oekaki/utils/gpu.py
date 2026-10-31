"""GPU sharing guard: inspect nvidia-smi before heavy work, never touch other users' processes."""
from __future__ import annotations

import os
import subprocess
import time


def gpu_status(index: int = 0) -> dict:
    """Free/total memory (MiB), utilisation (%) and compute processes on the GPU."""
    q = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    used, total, util = [float(x) for x in q.stdout.strip().split(",")]
    p = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    procs = []
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            procs.append(dict(pid=int(parts[0]), name=parts[1], mem=float(parts[2])))
    return dict(used=used, total=total, free=total - used, util=util, procs=procs)


def foreign_heavy_procs(status: dict, min_mem_mib: float = 2000) -> list[dict]:
    """Compute processes not started by us (different pid tree) that hold a lot of memory."""
    me = os.getpid()
    return [p for p in status["procs"] if p["pid"] != me and p["mem"] >= min_mem_mib]


def wait_for_gpu(required_mib: float, poll_s: float = 60, max_wait_s: float = 6 * 3600, log=print) -> dict:
    """Block until `required_mib` MiB are free. Returns the last status."""
    t0 = time.time()
    while True:
        st = gpu_status()
        if st["free"] >= required_mib:
            return st
        log(f"[gpu-guard] need {required_mib:.0f} MiB, free {st['free']:.0f} MiB "
            f"(procs: {[(p['pid'], p['name'], p['mem']) for p in st['procs']]}); waiting {poll_s}s")
        if time.time() - t0 > max_wait_s:
            raise RuntimeError("GPU did not become free in time")
        time.sleep(poll_s)


def ram_available_mib() -> float:
    """MemAvailable from /proc/meminfo (MiB)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return float("inf")


def wait_for_ram(required_mib: float, poll_s: float = 60, max_wait_s: float = 6 * 3600, log=print) -> float:
    """Block until `required_mib` MiB of host RAM are available (the machine is shared, avoid the OOM killer)."""
    t0 = time.time()
    while True:
        avail = ram_available_mib()
        if avail >= required_mib:
            return avail
        log(f"[ram-guard] need {required_mib:.0f} MiB host RAM, available {avail:.0f} MiB; waiting {poll_s}s")
        if time.time() - t0 > max_wait_s:
            raise RuntimeError("host RAM did not become available in time")
        time.sleep(poll_s)
