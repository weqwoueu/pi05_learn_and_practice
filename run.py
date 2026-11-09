"""用当前虚拟环境运行指定题的测试：python run.py 00 / all。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

TOPICS = {
    "00": "tensors", "01": "state", "02": "attention",
    "03": "transformer", "04": "flow", "05": "expert",
    "06": "finetune", "07": "rtc", "08": "value_advantage",
}

def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=[*TOPICS, "all"])
    parser.add_argument("-k", default=None, help="只运行名称匹配的测试")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    target = "tests" if args.exercise == "all" else f"tests/test_{args.exercise}_{TOPICS[args.exercise]}.py"
    command = [sys.executable, "-m", "pytest", "-q", "-x", target]
    if args.k:
        command += ["-k", args.k]
    print(f"Python: {sys.executable}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    raise SystemExit(subprocess.call(command, cwd=root, env=env))

if __name__ == "__main__":
    main()
