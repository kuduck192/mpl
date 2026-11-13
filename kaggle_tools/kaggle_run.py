#!/usr/bin/env python3
"""Ship sweeps of ssl_fmnist.py runs to Kaggle (GPU T4 x2) and fetch results.

  python kaggle_tools/kaggle_run.py push  sweeps/pilot.json      # build + push kernel
  python kaggle_tools/kaggle_run.py status <slug>
  python kaggle_tools/kaggle_run.py fetch  <slug>                # -> results/<slug>/

A sweep file is JSON: {"slug": "...", "jobs": [{"name": ..., "args": "--base fixmatch ..."}, ...],
                       "budget_hours": 11.0}
The kernel runs one worker per visible GPU; each worker pulls jobs from a shared
queue and launches `python ssl_fmnist.py <args>` with CUDA_VISIBLE_DEVICES set.
The Kaggle token is read by the kaggle CLI from ~/.kaggle/access_token (never
stored in this repository)."""
import base64
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _kaggle_user():
    """Kaggle username: $KAGGLE_USER or the untracked file .kaggle_user at the repo root."""
    u = os.environ.get("KAGGLE_USER")
    if not u and os.path.exists(os.path.join(ROOT, ".kaggle_user")):
        u = open(os.path.join(ROOT, ".kaggle_user")).read().strip()
    if not u:
        raise SystemExit("set KAGGLE_USER or write your Kaggle username to .kaggle_user")
    return u


USER = _kaggle_user()
BUILD = os.path.join(ROOT, ".kaggle_build")

DRIVER = r'''
import base64, json, os, queue, subprocess, sys, threading, time
T0 = time.time()
FILES = json.loads(base64.b64decode("{FILES_B64}").decode())
SWEEP = json.loads(base64.b64decode("{SWEEP_B64}").decode())
WORK = "/kaggle/working"
os.makedirs(WORK + "/results", exist_ok=True)
os.makedirs(WORK + "/logs", exist_ok=True)
for fname, b64 in FILES.items():
    with open(WORK + "/" + fname, "w") as f:
        f.write(base64.b64decode(b64).decode())
try:
    import torch
    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
except Exception as e:
    n_gpu, names = 0, [str(e)]
print("GPUs:", n_gpu, names, flush=True)
budget = float(SWEEP.get("budget_hours", 11.0)) * 3600
q = queue.Queue()
for j in SWEEP["jobs"]:
    q.put(j)
lock = threading.Lock()
summary = []

def worker(gid):
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        if time.time() - T0 > budget:
            print(f"[gpu{gid}] budget exhausted, skipping {j['name']}", flush=True)
            continue
        env = dict(os.environ)
        if n_gpu > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gid)
        remaining = (budget - (time.time() - T0)) / 60.0
        if j.get("est_min") and remaining < 1.1 * float(j["est_min"]):
            print(f"[gpu{gid}] {remaining:.0f} min left < 1.1 x est. {j['est_min']} min, skipping {j['name']}", flush=True)
            continue
        script = j.get("script", "ssl_fmnist.py")
        if script == "ssl_fmnist.py":
            cmd = [sys.executable, WORK + "/ssl_fmnist.py", "--name", j["name"], "--out_dir", WORK + "/results",
                   "--max_minutes", str(max(1.0, remaining - 2.0))] + j["args"].split()
        else:
            cmd = [sys.executable, WORK + "/" + script] + j["args"].split()
        t = time.time()
        with open(WORK + f"/logs/{j['name']}.log", "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        dt = (time.time() - t) / 60
        tail = ""
        try:
            with open(WORK + f"/logs/{j['name']}.log") as lf:
                tail = lf.read().strip().splitlines()[-1][:400]
        except Exception:
            pass
        with lock:
            summary.append({"name": j["name"], "rc": rc, "minutes": dt})
            print(f"[gpu{gid}] {j['name']} rc={rc} {dt:.1f} min :: {tail}", flush=True)
            with open(WORK + "/summary.json", "w") as f:
                json.dump(summary, f, indent=1)

threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, n_gpu))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ALL DONE in %.1f min" % ((time.time() - T0) / 60), flush=True)
'''


def build(sweep_path):
    with open(sweep_path) as f:
        sweep = json.load(f)
    slug = sweep["slug"]
    files = {}
    for rel in ["ssl_fmnist.py"] + sweep.get("files", []):
        with open(os.path.join(ROOT, rel), "rb") as f:
            files[os.path.basename(rel)] = base64.b64encode(f.read()).decode()
    files_b64 = base64.b64encode(json.dumps(files).encode()).decode()
    sweep_b64 = base64.b64encode(json.dumps(sweep).encode()).decode()
    d = os.path.join(BUILD, slug)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(DRIVER.replace("{FILES_B64}", files_b64).replace("{SWEEP_B64}", sweep_b64))
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["zalando-research/fashionmnist"],
        "competition_sources": [],
        "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return d, slug


def main():
    cmd = sys.argv[1]
    if cmd == "push":
        d, slug = build(sys.argv[2])
        r = subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"],
                           capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])
    elif cmd == "status":
        r = subprocess.run(["kaggle", "kernels", "status", f"{USER}/{sys.argv[2]}"], capture_output=True, text=True)
        print(r.stdout.strip(), r.stderr.strip()[-500:])
    elif cmd == "fetch":
        slug = sys.argv[2]
        out = os.path.join(ROOT, "results", slug)
        os.makedirs(out, exist_ok=True)
        r = subprocess.run(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", out, "-o"],
                           capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:])
    elif cmd == "logs":
        r = subprocess.run(["kaggle", "kernels", "logs", f"{USER}/{sys.argv[2]}"], capture_output=True, text=True)
        print(r.stdout[-6000:], r.stderr[-1000:])
    else:
        raise SystemExit("unknown command " + cmd)


if __name__ == "__main__":
    main()
