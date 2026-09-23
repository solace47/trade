"""Pass the macOS system HTTPS proxy to Python clients when needed."""

from __future__ import annotations

import os
import re
import subprocess
import sys


def effective_environment() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("HTTPS_PROXY") or env.get("https_proxy") or sys.platform != "darwin":
        return env
    result = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True)
    if result.returncode != 0:
        return env
    enabled = re.search(r"^\s*HTTPSEnable\s*:\s*(\d+)", result.stdout, re.MULTILINE)
    host = re.search(r"^\s*HTTPSProxy\s*:\s*(\S+)", result.stdout, re.MULTILINE)
    port = re.search(r"^\s*HTTPSPort\s*:\s*(\d+)", result.stdout, re.MULTILINE)
    if enabled and enabled.group(1) == "1" and host and port:
        env["HTTPS_PROXY"] = f"http://{host.group(1)}:{port.group(1)}"
    return env
