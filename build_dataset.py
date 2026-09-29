"""Copy 20 disjoint ChipBench zero-shot debugging cases into compact JSON sets."""

import json
import re
from pathlib import Path

from grade import grade

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "ChipBench" / "Verilog Debugging"
OUT = ROOT / "data"


def candidates(category: str) -> list[dict]:
    folder = SOURCE / f"dataset_debug_zero_shot_{category}"
    cases = []
    for prompt_file in sorted(folder.glob("*_prompt.txt")):
        stem = prompt_file.name.removesuffix("_prompt.txt")
        ref_file = folder / f"{stem}_ref.sv"
        test_file = folder / f"{stem}_test.sv"
        if not (ref_file.exists() and test_file.exists()):
            continue
        prompt = prompt_file.read_text(encoding="utf-8")
        blocks = re.findall(r"```(?:verilog|systemverilog)?\s*\n(.*?)```", prompt, re.S | re.I)
        if not blocks:
            continue
        cases.append({
            "id": stem,
            "category": category,
            "description": prompt.split("Based on the problem description above")[0].strip(),
            "buggy_code": blocks[-1].strip() + "\n",
            "correct_code": ref_file.read_text(encoding="utf-8"),
            "test_code": test_file.read_text(encoding="utf-8"),
            "source": str(prompt_file.relative_to(ROOT.parent)).replace("\\", "/"),
        })
    return cases


def main() -> None:
    categories = ["arithmetic", "assignment", "state_machine", "timing"]
    pools = {category: candidates(category) for category in categories}
    selected, used_problems = [], set()
    # Round-robin categories, avoid reusing a base problem, and keep only
    # cases whose supplied reference passes its own testbench.
    while len(selected) < 20:
        progressed = False
        for category in categories:
            while pools[category]:
                task = pools[category].pop(0)
                problem = task["id"].split("_", 1)[0]
                reference_as_dut = re.sub(r"\bmodule\s+RefModule\b", "module TopModule", task["correct_code"], count=1)
                if problem not in used_problems and grade(task, reference_as_dut)["passed"]:
                    used_problems.add(problem)
                    selected.append(task)
                    progressed = True
                    break
            if len(selected) == 20:
                break
        if not progressed:
            raise RuntimeError("Fewer than 20 distinct benchmark problems available")
    OUT.mkdir(exist_ok=True)
    for name, tasks in (("set_a", selected[:10]), ("set_b", selected[10:])):
        (OUT / f"{name}.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        print(name, len(tasks), [task["id"] for task in tasks])


if __name__ == "__main__":
    main()
