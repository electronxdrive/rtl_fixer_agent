"""Small file-backed source interface shared by both repair backends."""

import subprocess
import shutil
import tempfile
from pathlib import Path

from grade import grade

TOOL_LIMIT = 12
SIMULATION_LIMIT = 3


def simulation_quality(result: dict) -> tuple:
    """Compare candidates using simulator evidence only, including during eval."""
    mismatches = result.get("mismatches")
    return (bool(result.get("passed")), mismatches is not None,
            -mismatches if mismatches is not None else float("-inf"))


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
        if not self.tool_enabled(name):
            return False
        self.tool_calls += 1
        self.history.append(f"tool {self.tool_calls}/{TOOL_LIMIT}: {name}")
        return True

    def tool_enabled(self, name: str) -> bool:
        """Reserve the last call for simulation; test pending edits near the limit."""
        if not self.tools_available:
            return False
        if name in ("simulate", "simulate_fix"):
            return True
        return self.tool_calls < TOOL_LIMIT - 1 and not (
            self.tool_calls >= TOOL_LIMIT - 2 and self.code != self.last_simulated_code)

    def finish(self) -> None:
        """Never attach a report about rejected edits to the restored source."""
        reported_code = self.code
        self.verify()
        if self.code != reported_code:
            raise RuntimeError("Final edit failed validation and was reverted; repair report discarded")

    def reject(self, message: str) -> str:
        self.history.append(message)
        return message

    def trace(self) -> dict:
        # On an interrupted run, keep tested progress instead of an untested edit.
        if self.last_grade is not None and self.code != self.last_simulated_code:
            self.path.write_text(self.last_simulated_code, encoding="utf-8")
            self.history.append("exit: restored best tested candidate")
        code = self.code
        return {"fixed_code": code, "tool_calls": self.tool_calls,
                "initial_grade": getattr(self, "initial_grade", None),
                "tool_history": self.history, "simulations": self.simulations,
                "verified_grade": self.last_grade if code == self.last_simulated_code else None}

    def close(self) -> None:
        self.tmp.cleanup()

    def prepare(self) -> dict:
        """Anchor every backend/mode to the original RTL before any model edits."""
        self.last_simulated_code = self.task["buggy_code"]
        self.last_grade = grade(self.task, self.last_simulated_code)
        self.initial_grade = self.last_grade
        return self.verify()

    @property
    def code(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def search(self, pattern: str) -> str:
        """Search source with ripgrep and return numbered matches."""
        if not self.take_call("search"):
            return "Tools finished. Return the repair report."
        if not pattern or len(pattern) > 120:
            return "Provide a nonempty search pattern of at most 120 characters."
        # ripgrep may be on Codex's PATH but absent from the user's terminal.
        if not shutil.which("rg"):
            matches = [f"{i}: {line}" for i, line in enumerate(self.code.splitlines(), 1) if pattern in line]
            self.history.append(f"search {pattern!r}: {len(matches)} matches (Python fallback)")
            return "\n".join(matches[:30])[:4000] or "No matches."
        result = subprocess.run(
            ["rg", "-n", "-F", "--max-count", "30", "--", pattern, "dut.sv"],
            cwd=self.path.parent, capture_output=True, text=True, timeout=10,
        )
        self.history.append(f"search {pattern!r}: {len(result.stdout.splitlines())} matches")
        return result.stdout[:4000] or "No matches."

    def read_lines(self, start: int, end: int, original: bool = False) -> str:
        """Read at most 40 source lines, with original line numbers."""
        if not self.take_call("read"):
            return "Tools finished. Return the repair report."
        lines = (self.task["buggy_code"] if original else self.code).splitlines()
        if start < 1 or end < start:
            return "Choose a valid line range starting at 1 or above."
        end = min(end, start + 39)
        self.history.append(f"read {'original' if original else 'current'} lines {start}-{end}")
        return "\n".join(f"{number:4d} {lines[number - 1]}"
                         for number in range(start, min(end, len(lines)) + 1)) or "No lines in range."

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff to dut.sv using Git's patch implementation."""
        if not self.take_call("patch"):
            return "Tools finished. Return the repair report."
        if not shutil.which("git"):
            return self.reject("Git is unavailable. Use replace_source_line with line_count for this block.")
        headers = [line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))]
        if headers != ["--- a/dut.sv", "+++ b/dut.sv"]:
            return self.reject("Patch must edit only dut.sv with headers --- a/dut.sv and +++ b/dut.sv.")
        check = subprocess.run(["git", "apply", "--recount", "--check", "--include=dut.sv", "-"], input=patch,
                               cwd=self.path.parent, capture_output=True, text=True, timeout=10)
        if check.returncode:
            self.history.append("patch rejected")
            return self.reject(f"Patch rejected: {check.stderr[-1000:]}")
        before = self.code
        result = subprocess.run(["git", "apply", "--recount", "--include=dut.sv", "-"], input=patch,
                                cwd=self.path.parent, capture_output=True, text=True, timeout=10)
        if result.returncode:
            self.history.append("patch failed")
            return self.reject(f"Patch failed: {result.stderr[-1000:]}")
        self.edits += self.code != before
        self.history.append("patch applied" if self.code != before else "patch made no change")
        return "Patch applied." if self.code != before else "Patch made no change."

    def replace_line(self, line_number: int, new_line: str, line_count: int = 1) -> str:
        """Replace a short numbered range in one call; default is one line."""
        if not self.take_call("replace_line"):
            return "Tools finished. Return the repair report."
        lines = self.code.splitlines(keepends=True)
        replacement_lines = new_line.splitlines()
        if (line_number < 1 or not 1 <= line_count <= 40 or line_number + line_count - 1 > len(lines)
                or len(replacement_lines) > 40):
            return self.reject("Choose an existing range and replacement of at most 40 lines.")
        original = "".join(lines[line_number - 1:line_number - 1 + line_count])
        replacement = "\n".join(replacement_lines) + ("\n" if original.endswith("\n") and replacement_lines else "")
        if replacement == original:
            self.history.append(f"line {line_number} unchanged")
            return "Line unchanged."
        lines[line_number - 1:line_number - 1 + line_count] = [replacement]
        self.path.write_text("".join(lines), encoding="utf-8")
        self.edits += 1
        self.history.append(f"replaced lines {line_number}-{line_number + line_count - 1}")
        return f"Updated line {line_number}: {new_line}"

    def simulate(self) -> dict:
        if not self.take_call("simulate"):
            return {"error": "Tools finished. Return the repair report."}
        self.simulation_calls += 1
        return self.verify()

    def verify(self) -> dict:
        """Validate current RTL and retain the best tested source, including at exit.

        Initial/final runner checks do not consume the agent's tool budget.
        """
        current_code = self.code
        if current_code == self.last_simulated_code:
            return self.last_grade
        outcome = grade(self.task, current_code)
        self.simulations.append(outcome)
        # Undo a tested regression so the next edit starts from the better RTL.
        if self.last_grade is not None and simulation_quality(outcome) < simulation_quality(self.last_grade):
            self.path.write_text(self.last_simulated_code, encoding="utf-8")
            self.history.append("simulation regressed; restored previous tested candidate")
            return {**self.last_grade, "rejected_trial": outcome,
                    "note": "Restored the previous better candidate in dut.sv. Revise that source."}
        self.last_simulated_code = current_code
        self.last_grade = outcome
        self.history.append(f"simulate: {outcome.get('mismatches')} mismatches, passed={outcome['passed']}")
        return outcome
