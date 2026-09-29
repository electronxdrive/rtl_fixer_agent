"""Small file-backed source interface shared by both repair backends."""

import subprocess
import tempfile
from pathlib import Path

from grade import grade

TOOL_LIMIT = 12
SIMULATION_LIMIT = 3


class RepairError(RuntimeError):
    """Keep the partial repair and measured usage when an agent stops early."""

    def __init__(self, message: str, trace: dict):
        super().__init__(message)
        self.trace = trace


class SourceWorkspace:
    def __init__(self, task: dict, starting_code: str | None = None):
        self.task = task
        self.tmp = tempfile.TemporaryDirectory(prefix="silicon_source_")
        self.path = Path(self.tmp.name) / "dut.sv"
        self.path.write_text(starting_code if starting_code is not None else task["buggy_code"], encoding="utf-8")
        self.simulations = []
        self.last_simulated_code = None
        self.last_grade = None
        self.edits = 0
        self.history = []
        self.tool_calls = 0
        self.simulation_calls = 0

    @property
    def tools_available(self) -> bool:
        return (self.tool_calls < TOOL_LIMIT and self.simulation_calls < SIMULATION_LIMIT
                and not (self.last_grade or {}).get("passed", False))

    def take_call(self, name: str) -> bool:
        """Enforce one shared budget for either backend, including invalid calls."""
        if not self.tools_available:
            return False
        self.tool_calls += 1
        self.history.append(f"tool {self.tool_calls}/{TOOL_LIMIT}: {name}")
        return True

    def trace(self) -> dict:
        code = self.code
        return {"fixed_code": code, "tool_calls": self.tool_calls,
                "tool_history": self.history, "simulations": self.simulations,
                "verified_grade": self.last_grade if code == self.last_simulated_code else None}

    def close(self) -> None:
        self.tmp.cleanup()

    @property
    def code(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def search(self, pattern: str) -> str:
        """Search source with ripgrep and return numbered matches."""
        if not self.take_call("search"):
            return "Tools finished. Return the repair report."
        if not pattern or len(pattern) > 120:
            return "Provide a nonempty search pattern of at most 120 characters."
        result = subprocess.run(
            ["rg", "-n", "-F", "--max-count", "30", "--", pattern, "dut.sv"],
            cwd=self.path.parent, capture_output=True, text=True, timeout=10,
        )
        self.history.append(f"search {pattern!r}: {len(result.stdout.splitlines())} matches")
        return result.stdout[:4000] or "No matches."

    def read_lines(self, start: int, end: int) -> str:
        """Read at most 40 source lines, with original line numbers."""
        if not self.take_call("read"):
            return "Tools finished. Return the repair report."
        lines = self.code.splitlines()
        if start < 1 or end < start or end - start >= 40:
            return "Choose a valid range of at most 40 lines."
        self.history.append(f"read lines {start}-{end}")
        return "\n".join(f"{number:4d} {lines[number - 1]}"
                         for number in range(start, min(end, len(lines)) + 1)) or "No lines in range."

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff to dut.sv using Git's patch implementation."""
        if not self.take_call("patch"):
            return "Tools finished. Return the repair report."
        headers = [line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))]
        if headers != ["--- a/dut.sv", "+++ b/dut.sv"]:
            return "Patch must edit only dut.sv with headers --- a/dut.sv and +++ b/dut.sv."
        check = subprocess.run(["git", "apply", "--check", "--include=dut.sv", "-"], input=patch,
                               cwd=self.path.parent, capture_output=True, text=True, timeout=10)
        if check.returncode:
            self.history.append("patch rejected")
            return f"Patch rejected: {check.stderr[-1000:]}"
        before = self.code
        result = subprocess.run(["git", "apply", "--include=dut.sv", "-"], input=patch,
                                cwd=self.path.parent, capture_output=True, text=True, timeout=10)
        if result.returncode:
            self.history.append("patch failed")
            return f"Patch failed: {result.stderr[-1000:]}"
        self.edits += self.code != before
        self.history.append("patch applied" if self.code != before else "patch made no change")
        return "Patch applied." if self.code != before else "Patch made no change."

    def replace_line(self, line_number: int, new_line: str) -> str:
        """Replace one numbered line, preserving the file's line endings."""
        if not self.take_call("replace_line"):
            return "Tools finished. Return the repair report."
        lines = self.code.splitlines(keepends=True)
        if line_number < 1 or line_number > len(lines) or "\n" in new_line or "\r" in new_line:
            return "Choose an existing line number and one replacement line without a newline."
        original = lines[line_number - 1]
        ending = "\r\n" if original.endswith("\r\n") else "\n" if original.endswith("\n") else ""
        replacement = new_line + ending
        if replacement == original:
            self.history.append(f"line {line_number} unchanged")
            return "Line unchanged."
        lines[line_number - 1] = replacement
        self.path.write_text("".join(lines), encoding="utf-8")
        self.edits += 1
        self.history.append(f"replaced line {line_number}")
        return f"Updated line {line_number}: {new_line}"

    def simulate(self) -> dict:
        if not self.take_call("simulate"):
            return {"error": "Tools finished. Return the repair report."}
        self.simulation_calls += 1
        current_code = self.code
        if current_code == self.last_simulated_code:
            return self.last_grade
        outcome = grade(self.task, current_code)
        self.last_simulated_code = current_code
        self.last_grade = outcome
        self.simulations.append(outcome)
        self.history.append(f"simulate: {outcome.get('mismatches')} mismatches, passed={outcome['passed']}")
        return outcome
