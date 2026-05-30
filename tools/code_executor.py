"""
MCP Tool: code_executor
Simulates safe command execution for SRE diagnostics — runs pre-approved
diagnostic commands and returns mock or real output. In production this
would be sandboxed; here it executes a curated safe-list.
"""

from __future__ import annotations

import subprocess
import shlex
from typing import Any, Dict

from tools.base import MCPTool, register_tool


# Commands that are allowed to run in simulation mode
_SAFE_COMMANDS = {
    "uptime", "df", "free", "vmstat", "iostat", "top", "ps",
    "netstat", "ss", "uname", "hostname", "date", "whoami",
}


@register_tool
class CodeExecutorTool(MCPTool):
    name = "code_executor"
    description = (
        "Execute safe diagnostic shell commands (e.g. `df -h`, `free -m`). "
        "Only a curated safe-list of read-only commands is permitted."
    )
    input_schema = {
        "command": "str — the shell command to execute",
        "simulate": "bool — if True, return canned output without running (default True)",
    }

    def execute(self, command: str, simulate: bool = True) -> Dict[str, Any]:
        tokens = shlex.split(command)
        if not tokens:
            return {"error": "Empty command"}

        base_cmd = tokens[0]
        if base_cmd not in _SAFE_COMMANDS:
            return {
                "error": f"Command '{base_cmd}' is not in the safe-list",
                "allowed": sorted(_SAFE_COMMANDS),
            }

        if simulate:
            return {"command": command, "stdout": _SIMULATED.get(base_cmd, "(no simulation available)"), "returncode": 0}

        try:
            result = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out", "command": command}
        except Exception as exc:
            return {"error": str(exc), "command": command}


_SIMULATED: Dict[str, str] = {
    "uptime": " 14:32:01 up 12 days,  3:47,  2 users,  load average: 4.52, 4.11, 3.98",
    "free": (
        "              total        used        free      shared  buff/cache   available\n"
        "Mem:       16384000    14200000      384000      102400     1800000      881600\n"
        "Swap:       8192000     4096000     4096000"
    ),
    "df": (
        "Filesystem      Size  Used Avail Use% Mounted on\n"
        "/dev/sda1        50G   47G  1.2G  98% /\n"
        "tmpfs           7.8G     0  7.8G   0% /dev/shm"
    ),
    "vmstat": (
        "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n"
        " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st\n"
        " 6  2 4096000 384000  20000 1800000  100  200   500  1000 3000 5000 85  5  5  5  0"
    ),
}
