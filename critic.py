"""Reference-aware RCA grader; its feedback is used only during training."""

import json

from agents import Agent, Runner
from pydantic import BaseModel, Field

from scorecard import changed_regions, reference_as_dut


class Critique(BaseModel):
    cause_score: float = Field(ge=0, le=1, description="Semantic correctness of the attempted root cause")
    feedback: str = Field(description="Specific next-step guidance for the fixer")
    intended_behavior: str = Field(description="Required RTL behavior and timing")
    parameters_to_check: str = Field(description="Relevant widths, reset, clock, state, counts, or handshake values")
    diagnostic_cue: str = Field(description="What to inspect for this bug pattern")
    root_cause: str = Field(description="Root cause supported by the training reference")
    evidence: str = Field(description="Why the buggy behavior differs from the required behavior")
    fix_summary: str = Field(description="Concise reusable repair rule")


def review(task: dict, attempt: dict, model: str) -> tuple[Critique, dict]:
    # Cover every defect, using each file's own line coordinates.
    def snippet(source: str, start: int, end: int) -> str:
        lines = source.splitlines()
        return "\n".join(f"{number:4d} {lines[number - 1]}"
                         for number in range(max(1, start - 6), min(len(lines), end + 6) + 1))

    regions = changed_regions(task)
    prompt = json.dumps({
        "specification": task["description"],
        "changed_regions": [{
            "original_lines": [start, end],
            "buggy_snippet": snippet(task["buggy_code"], start, end),
            "reference_snippet": snippet(reference_as_dut(task), ref_start, ref_end),
            "attempted_snippet": snippet(attempt["fixed_code"], start, end),
        } for start, end, ref_start, ref_end in regions],
        "attempt": {key: attempt.get(key) for key in (
            "root_cause", "bug_line", "intended_behavior", "parameters_to_check", "diagnostic_cue",
            "evidence", "fix_summary", "grade", "error")},
    })
    agent = Agent(
        name="RTL root-cause grader",
        model=model,
        instructions=("Review the attempted RTL repair against the specification, "
                      "numbered source snippets, and simulator result. Identify the exact "
                      "remaining mistake. Grade the attempted root cause by meaning: "
                      "1.0 for a correct causal explanation even if wording differs "
                      "from the RTL, 0.0 for an incorrect or missing explanation, "
                      "and partial credit for incomplete explanations. Check every changed "
                      "region: an explanation covering only one of multiple defects is incomplete. "
                      "The reported primary bug line may identify any real defect. Give a short "
                      "correction and a reusable diagnostic lesson with intended "
                      "behavior and concrete parameters to check. If the reported "
                      "bug_line is wrong, state the correct original source line in "
                      "feedback even when the RTL passes simulation. Do not output a long trace."),
        output_type=Critique,
    )
    result = Runner.run_sync(agent, prompt, max_turns=2)
    usage = result.context_wrapper.usage
    details = getattr(usage, "input_tokens_details", None)
    return result.final_output, {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": getattr(details, "cached_tokens", 0) or 0,
        "output_tokens": usage.output_tokens,
        "requests": usage.requests,
    }
