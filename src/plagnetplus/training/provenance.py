from pathlib import Path
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

def save_run_record(output, *, model, manifest, seed, config):
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "seed": seed,
        "git_commit": git_commit(),
        "manifest_sha256": sha256_file(manifest),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "config": config,
    }
    Path(output).write_text(json.dumps(record, indent=2), encoding="utf-8")
