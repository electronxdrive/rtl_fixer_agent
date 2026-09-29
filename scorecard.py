"""Three diagnostic scores from the supplied architecture; simulation decides pass."""

import re
from difflib import SequenceMatcher

def reference_as_dut(task: dict) -> str:
    return re.sub(r"\bmodule\s+RefModule\b", "module TopModule", task["correct_code"], count=1)


def changed_regions(task: dict) -> list[tuple[int, int, int, int]]:
    """Return every changed region in original and reference line coordinates."""
    def significant_lines(code: str) -> list[tuple[int, str]]:
        code = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), code, flags=re.S)
        return [(number, re.sub(r"\s+", "", line))
                for number, line in enumerate(code.splitlines(), 1)
                if line.strip() and not line.lstrip().startswith("`timescale")]

    buggy = significant_lines(task["buggy_code"])
    correct = significant_lines(reference_as_dut(task))
    match = SequenceMatcher(None, [line for _, line in buggy],
                            [line for _, line in correct], autojunk=False)
    def bounds(lines, start, end):
        if not lines:
            return 1, 1
        return lines[min(start, len(lines) - 1)][0], lines[min(max(start, end - 1), len(lines) - 1)][0]

    return [(*bounds(buggy, i, j), *bounds(correct, k, l))
            for op, i, j, k, l in match.get_opcodes() if op != "equal"]


def scorecard(task: dict, fixed_code: str, cause_score: float, bug_line: int | None) -> dict:
    # These are diagnostics. The simulator in grade.py is the pass criterion.
    regions = changed_regions(task)
    # Comments and whitespace do not change RTL behavior or repair quality.
    def normalize(source: str) -> str:
        without_comments = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
        return re.sub(r"\s+", "", without_comments)
    similarity = (0.0 if normalize(fixed_code) == normalize(task["buggy_code"]) else
                  SequenceMatcher(None, normalize(fixed_code),
                                  normalize(reference_as_dut(task)), autojunk=False).ratio())
    cause_overlap = max(0.0, min(1.0, cause_score))
    line_proximity = int(bug_line is not None and any(
        start - 2 <= bug_line <= end + 2 for start, end, _, _ in regions))
    return {
        "code_similarity": round(similarity, 3),
        "cause_overlap": round(cause_overlap, 3),
        "bug_line_proximity": line_proximity,
        "score": round(0.5 * min(similarity / 0.85, 1) + 0.3 * cause_overlap + 0.2 * line_proximity, 3),
    }
