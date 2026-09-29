"""Execution grader using ChipBench's supplied Icarus Verilog testbenches."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

IVERILOG = Path(shutil.which("iverilog") or r"C:\iverilog\bin\iverilog.exe")
VVP = Path(shutil.which("vvp") or r"C:\iverilog\bin\vvp.exe")


def grade(task: dict, fixed_code: str) -> dict:
    if not (IVERILOG.exists() and VVP.exists()):
        raise RuntimeError("Icarus Verilog is required for benchmark grading")
    with tempfile.TemporaryDirectory(prefix="silicon_grade_") as tmp:
        path = Path(tmp)
        (path / "dut.sv").write_text(fixed_code, encoding="utf-8")
        (path / "ref.sv").write_text(task["correct_code"], encoding="utf-8")
        (path / "test.sv").write_text(task["test_code"], encoding="utf-8")
        try:
            compile_run = subprocess.run(
                [str(IVERILOG), "-g2012", "-o", "sim.vvp", "dut.sv", "ref.sv", "test.sv"],
                cwd=path, capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "mismatches": None, "error": "compiler timeout"}
        if compile_run.returncode:
            return {"passed": False, "mismatches": None, "error": compile_run.stderr[-2000:]}
        try:
            sim = subprocess.run([str(VVP), "sim.vvp"], cwd=path, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {"passed": False, "mismatches": None, "error": "simulation timeout"}
        # ChipBench prints a final mismatch count; zero with tested samples is a pass.
        matches = re.findall(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples", sim.stdout)
        if sim.returncode or not matches or "TIMEOUT" in sim.stdout:
            return {"passed": False, "mismatches": None, "error": (sim.stderr + sim.stdout)[-2000:]}
        mismatches, samples = map(int, matches[-1])
        # Supplied benches often report which output first diverged. These
        # short hints help the fixer without exposing the reference source.
        hints = re.findall(r"^Hint:.*$", sim.stdout, flags=re.M)[:8]
        return {"passed": mismatches == 0 and samples > 0,
                "mismatches": mismatches, "samples": samples, "hints": hints}
