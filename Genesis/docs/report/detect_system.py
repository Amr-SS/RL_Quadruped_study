"""
Print the host's CPU / GPU / RAM / OS so the README hardware table can be
reproduced or refreshed on a different machine. Read-only; no side effects.

Run:  python docs/report/detect_system.py
"""

import platform
import shutil
import subprocess


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main():
    print("== CPU ==")
    model = sh(r"lscpu | sed -n 's/^Model name:\s*//p'") or platform.processor()
    cores = sh(r"lscpu | sed -n 's/^Core(s) per socket:\s*//p'")
    threads = sh(r"lscpu | sed -n 's/^CPU(s):\s*//p'")
    print(f"  {model}  ({cores} cores / {threads} threads)" if cores else f"  {model}")

    print("== GPU ==")
    if shutil.which("nvidia-smi"):
        print("  " + (sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                         "--format=csv,noheader") or "nvidia-smi query failed"))
    else:
        print("  no NVIDIA GPU detected")

    print("== RAM ==")
    print("  " + (sh("free -h | awk '/^Mem:/{print $2\" total\"}'") or "unknown"))

    print("== OS ==")
    pretty = sh(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME")
    print(f"  {pretty or platform.system()}  |  kernel {platform.release()}")

    print("== Python ==")
    print(f"  {platform.python_version()}")


if __name__ == "__main__":
    main()
