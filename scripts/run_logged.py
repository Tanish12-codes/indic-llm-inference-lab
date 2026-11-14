"""Run a command, show its output live, and save a UTF-8 copy to results/logs/<name>.log.
(PowerShell's Tee-Object writes UTF-16 and Windows consoles choke on Devanagari, so we do it here.)
Usage: python scripts/run_logged.py <log-name> <command...>
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
logs = ROOT / "results" / "logs"
logs.mkdir(parents=True, exist_ok=True)
name, cmd = sys.argv[1], sys.argv[2:]
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
with open(logs / f"{name}.log", "w", encoding="utf-8") as f:
    f.write("$ " + " ".join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for raw in p.stdout:
        line = raw.decode("utf-8", errors="replace")
        f.write(line)
        f.flush()
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            sys.stdout.write(line.encode("ascii", "replace").decode())
        sys.stdout.flush()
    rc = p.wait()
    f.write(f"\n[exit code {rc}]\n")
sys.exit(rc)
