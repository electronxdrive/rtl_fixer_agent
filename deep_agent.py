"""Optional Deep Agents backend using its built-in agent loop and structured output."""

from deepagents import create_deep_agent
from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from langchain.agents.middleware import wrap_model_call
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from agent import Repair, RepairReport, SYSTEM, build_prompt
from source_workspace import RepairError, SourceWorkspace, TOOL_LIMIT


class Usage(UsageMetadataCallbackHandler):
    """The package tracks tokens even if the graph later fails."""

    requests = 0

    def on_llm_end(self, response, **kwargs):
        super().on_llm_end(response, **kwargs)
        self.requests += 1

    def counts(self) -> dict:
        items = list(self.usage_metadata.values())
        return {"input_tokens": sum(i.get("input_tokens", 0) for i in items),
                "output_tokens": sum(i.get("output_tokens", 0) for i in items),
                "cached_input_tokens": sum(i.get("input_token_details", {}).get("cache_read", 0) for i in items),
                "requests": self.requests}


def repair_deep(task: dict, hints: list[dict], feedback: str, model: str,
                starting_code: str | None = None) -> tuple[Repair, dict]:
    prompt = build_prompt(task, hints, feedback)
    workspace = SourceWorkspace(task, starting_code)
    usage = Usage()

    @wrap_model_call
    def tool_budget(request, handler):
        if usage.requests >= TOOL_LIMIT + 2:
            raise RuntimeError("Model request budget exhausted")
        return handler(request.override(
            tools=request.tools if workspace.tools_available else [],
            model_settings={**request.model_settings, "parallel_tool_calls": False}))

    @tool
    def search_source(pattern: str) -> str:
        """Find literal text in dut.sv with ripgrep and numbered matches."""
        return workspace.search(pattern)

    @tool
    def read_source_lines(start: int, end: int) -> str:
        """Read at most 40 numbered source lines, inclusive."""
        return workspace.read_lines(start, end)

    @tool
    def apply_source_patch(patch: str) -> str:
        """Apply a unified diff to dut.sv using Git."""
        return workspace.apply_patch(patch)

    @tool
    def replace_source_line(line_number: int, new_line: str) -> str:
        """Replace one numbered source line with the complete corrected line."""
        return workspace.replace_line(line_number, new_line)

    @tool
    def simulate_fix() -> dict:
        """Compile and simulate the current dut.sv with its testbench."""
        return workspace.simulate()

    try:
        # Keep the Deep Agents loop, exposing only the five RTL tools.
        register_harness_profile(f"openai:{model}", HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            excluded_tools=frozenset({"ls", "read_file", "write_file", "edit_file", "delete",
                                      "glob", "grep", "execute", "task", "write_todos"})))
        agent = create_deep_agent(
            model=f"openai:{model}", system_prompt=SYSTEM,
            tools=[search_source, read_source_lines, replace_source_line,
                   apply_source_patch, simulate_fix],
            middleware=[tool_budget], response_format=RepairReport)
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                              config={"recursion_limit": 6 * (TOOL_LIMIT + 2), "callbacks": [usage]})
        if workspace.code == task["buggy_code"]:
            raise RuntimeError("Fixer identified a bug but did not edit dut.sv")
        answer = Repair(**result["structured_response"].model_dump(), fixed_code=workspace.code)
        return answer, {
            **workspace.trace(), **usage.counts(), "prompt": prompt,
            "usage_incomplete": not bool(usage.usage_metadata),
        }
    except Exception as exc:
        raise RepairError(f"{type(exc).__name__}: {exc}", {
            **workspace.trace(), **usage.counts(), "prompt": prompt,
            "usage_incomplete": not isinstance(exc, GraphRecursionError) or not bool(usage.usage_metadata),
        }) from exc
    finally:
        workspace.close()
