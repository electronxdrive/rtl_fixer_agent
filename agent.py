"""One focused RTL repair agent built on the OpenAI Agents SDK."""

import json

from agents import Agent, ModelSettings, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded
from pydantic import BaseModel, Field

from source_workspace import RepairError, SourceWorkspace, TOOL_LIMIT


class RepairReport(BaseModel):
    intended_behavior: str = Field(description="What the RTL is supposed to do, including timing and state behavior")
    parameters_to_check: str = Field(description="Relevant widths, bit order, reset, clock, state, counts, or handshake conditions and their expected values")
    root_cause: str = Field(description="Concise explanation of the actual RTL defect")
    bug_line: int = Field(description="One-indexed primary faulty line in the original buggy source")
    diagnostic_cue: str = Field(description="What signal, timing, or condition to inspect for this bug pattern")
    evidence: str = Field(description="How the faulty line contradicts the specification or simulation")
    fix_summary: str = Field(description="What was changed and why it repairs the behavior")


class Repair(RepairReport):
    fixed_code: str


SYSTEM = """You repair a buggy Verilog/SystemVerilog TopModule.
Use search_source and read_source_lines to inspect relevant numbered snippets.
Be economical with tools: inspect the smallest relevant snippets, then edit.
Aim for at most four searches/reads before the first edit and at most three
simulations. The hard budget is 12 tool calls and 3 simulation calls per attempt.
Tools are disabled after a passing simulation or an exhausted budget; then
return your report with the evidence available. Do not repeat unchanged reads.
Do not scan the whole file and then read the same original source again.
Original snippets are needed only where prior edits changed the relevant lines.
Prefer the smallest evidenced correction to a rewrite. Preserve unrelated logic.
Reserve calls for simulate_fix: test your first repair before spending the last
two tool calls. Near the limit, only simulate_fix is available until pending
edits are tested; the last tool call is reserved for simulation.
If an edit is rejected, correct the tool arguments rather than
repeatedly submitting an oversized replacement. Report only changes actually
present in the retained source, not an intended rewrite that was never completed.
Fix dut.sv before returning a repair report, unless the carried-over repair
already passes and only its explanation needs correction. For a one-line defect,
call replace_source_line(line_number, new_line) with the complete corrected line.
For a short block, set line_count to the number of existing lines to replace
and new_line to the complete replacement text. Prefer this to several single-line
calls. For a larger patch, call apply_source_patch with a unified diff using only
dut.sv and headers --- a/dut.sv and +++ b/dut.sv. Call simulate_fix after editing,
then revise based on simulator feedback if needed. If simulation passes, stop
using tools and return the repair report immediately. Pay particular attention to
reset polarity, clock edges, nonblocking assignments, bit widths, and state
transitions when relevant. Do not redesign unrelated behavior.
Keep the module interface and name. Return the one-indexed original bug line,
the intended behavior, specific parameters to check, a diagnostic cue, a brief
root cause, concrete evidence, and how the fix restores the intended behavior.
These short fields form a reviewable reasoning summary for the knowledge base.
State concrete values and conditions where the specification provides them.
Do not mention or assume access to hidden reference code.
"""


def build_prompt(task: dict, hints: list[dict], feedback: str) -> str:
    lesson_block = "\n\n".join(h["lesson"] for h in hints) if hints else "(none)"
    return f"""Specification:\n{task['description']}\n\nSource file: dut.sv
Use search_source(pattern) or read_source_lines(start, end) to inspect it.
Use replace_source_line(line_number, new_line) for one-line changes or
apply_source_patch(patch) for a multi-line change, then simulate_fix().
The source may contain the previous attempt's repair. Feedback describes that
candidate. Continue from it. Report bug_line in the original source numbering.
Only read original=True snippets if the relevant lines have changed and you need
the original defect's line number. Do not reread unchanged original snippets.
Correct only evidenced defects; avoid redesigning other logic.
\nRelevant RCA lessons:\n{lesson_block}
\nPrevious attempt feedback:\n{feedback or '(first attempt)'}
"""


def usage_counts(usage) -> dict:
    """Use the SDK's accumulated usage for successful and interrupted runs."""
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cached_input_tokens": getattr(usage.input_tokens_details, "cached_tokens", 0) or 0,
            "requests": usage.requests}


def repair(task: dict, hints: list[dict], feedback: str, model: str,
           starting_code: str | None = None) -> tuple[Repair, dict]:
    prompt = build_prompt(task, hints, feedback)
    workspace = SourceWorkspace(task, starting_code)
    enabled = lambda context, agent: workspace.tool_enabled("read")

    @function_tool(is_enabled=enabled)
    def search_source(pattern: str) -> str:
        """Find literal source text with ripgrep and return line-numbered matches."""
        return workspace.search(pattern)

    @function_tool(is_enabled=enabled)
    def read_source_lines(start: int, end: int, original: bool = False) -> str:
        """Read up to 40 numbered lines. original=True reads the original buggy RTL."""
        return workspace.read_lines(start, end, original)

    @function_tool(is_enabled=enabled)
    def apply_source_patch(patch: str) -> str:
        """Apply a unified diff to dut.sv with Git. Read snippets first."""
        return workspace.apply_patch(patch)

    @function_tool(is_enabled=enabled)
    def replace_source_line(line_number: int, new_line: str, line_count: int = 1) -> str:
        """Replace line_count existing lines with new_line text; each is capped at 40 lines."""
        return workspace.replace_line(line_number, new_line, line_count)

    @function_tool(is_enabled=lambda context, agent: workspace.tool_enabled("simulate"))
    def simulate_fix() -> str:
        """Compile and run the current dut.sv against the supplied testbench."""
        return json.dumps(workspace.simulate())

    agent = Agent(name="RTL fixer", model=model, instructions=SYSTEM,
                  model_settings=ModelSettings(parallel_tool_calls=False),
                  tools=[search_source, read_source_lines, replace_source_line,
                         apply_source_patch, simulate_fix],
                  output_type=RepairReport)
    result = None
    try:
        initial = workspace.prepare()
        prompt += "\nStarting candidate simulator result: " + json.dumps(initial)
        # Reserve a final reporting turn after the tool budget is exhausted.
        result = Runner.run_sync(agent, prompt, max_turns=TOOL_LIMIT + 2)
        workspace.finish()
        return Repair(**result.final_output.model_dump(), fixed_code=workspace.code), {
            **workspace.trace(), **usage_counts(result.context_wrapper.usage), "prompt": prompt,
        }
    except Exception as exc:
        data = result or getattr(exc, "run_data", None)
        trace = {**workspace.trace(), "prompt": prompt,
                 "usage_incomplete": data is None or (result is None and not isinstance(exc, MaxTurnsExceeded))}
        if data is not None:
            trace.update(usage_counts(data.context_wrapper.usage))
        raise RepairError(f"{type(exc).__name__}: {exc}", trace) from exc
    finally:
        workspace.close()
