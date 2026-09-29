"""Offline regressions for grading, retry state, budgets, and usage accounting."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
import critic
import deep_agent
import run
from agents.tool_context import ToolContext
from agents.exceptions import MaxTurnsExceeded
from scorecard import changed_regions, scorecard
from source_workspace import RepairError, SourceWorkspace, TOOL_LIMIT

TASK = {"id": "test", "category": "timing", "description": "Fix the output.",
        "buggy_code": "module TopModule; endmodule\n"}
FIXED = "module TopModule; wire x; endmodule\n"
PASS = {"passed": True, "mismatches": 0, "samples": 10}
FAIL = {"passed": False, "mismatches": 2, "samples": 10}


class PipelineTests(unittest.TestCase):
    def test_every_changed_region_is_graded(self):
        tasks = json.loads((run.ROOT / "data/set_b.json").read_text())
        task = next(t for t in tasks if t["id"].startswith("Prob033"))
        self.assertEqual([r[0] for r in changed_regions(task)], [44, 54])
        self.assertEqual(scorecard(task, task["correct_code"], 1, 54)["bug_line_proximity"], 1)
        with patch.object(critic.Runner, "run_sync", side_effect=RuntimeError("offline")) as runner:
            with self.assertRaises(RuntimeError):
                critic.review(task, {"fixed_code": task["buggy_code"]}, "gpt-5.5")
        regions = json.loads(runner.call_args.args[1])["changed_regions"]
        self.assertEqual([r["original_lines"][0] for r in regions], [44, 54])
        self.assertIn("state <= s3_yellow", regions[1]["reference_snippet"])

    def test_unscored_tasks_are_not_dropped(self):
        self.assertEqual(run.average_scores([{"scorecard": {"score": 1}}, {}])["score"], 0.5)

    def test_budget_prevents_further_edits(self):
        w = SourceWorkspace(TASK)
        try:
            for _ in range(TOOL_LIMIT):
                w.read_lines(1, 1)
            self.assertFalse(w.tools_available)
            w.replace_line(1, FIXED.strip())
            self.assertEqual(w.code, TASK["buggy_code"])
        finally:
            w.close()

    def test_simulation_cache_and_stop_on_pass(self):
        w = SourceWorkspace(TASK)
        try:
            with patch("source_workspace.grade", side_effect=[FAIL, PASS]) as grader:
                w.simulate()
                w.simulate()
                self.assertEqual(grader.call_count, 1)
                w.replace_line(1, FIXED.strip())
                w.simulate()
                self.assertEqual(grader.call_count, 2)
                self.assertFalse(w.tools_available)
                w.replace_line(1, "discarded edit")
                self.assertEqual(w.code, FIXED)
                self.assertEqual(w.trace()["verified_grade"], PASS)
        finally:
            w.close()

    def test_sdk_failure_keeps_partial_usage_and_candidate(self):
        usage = SimpleNamespace(input_tokens=100, output_tokens=10, requests=3,
                                input_tokens_details=SimpleNamespace(cached_tokens=20))

        def fail_after_tools(fixer, prompt, **kwargs):
            tools = {tool.name: tool for tool in fixer.tools}
            for name, arguments in [("replace_source_line", {"line_number": 1, "new_line": FIXED.strip()}),
                                    ("simulate_fix", {})]:
                arguments = json.dumps(arguments)
                ctx = ToolContext(context=None, tool_name=name, tool_call_id=name, tool_arguments=arguments)
                asyncio.run(tools[name].on_invoke_tool(ctx, arguments))
            exc = MaxTurnsExceeded("test limit")
            exc.run_data = SimpleNamespace(context_wrapper=SimpleNamespace(usage=usage))
            raise exc

        with patch.object(agent.Runner, "run_sync", side_effect=fail_after_tools), \
                patch("source_workspace.grade", return_value=PASS):
            with self.assertRaises(RepairError) as error:
                agent.repair(TASK, [], "", "gpt-5.5")
        trace = error.exception.trace
        self.assertEqual(trace["fixed_code"], FIXED)
        self.assertEqual(trace["verified_grade"], PASS)
        self.assertEqual(trace["input_tokens"], 100)
        self.assertEqual(trace["cached_input_tokens"], 20)
        self.assertFalse(trace["usage_incomplete"])

    def test_error_record_preserves_simulation_and_usage(self):
        trace = {"fixed_code": FIXED, "verified_grade": PASS, "input_tokens": 100,
                 "output_tokens": 10, "usage_incomplete": False}
        with patch.object(run, "repair", side_effect=RepairError("limit", trace)):
            record = run.one(TASK, None, "gpt-5.5")
        self.assertTrue(record["passed"])
        self.assertEqual(record["grade"], PASS)
        self.assertGreater(record["estimated_model_cost_usd"], 0)
        self.assertEqual(run.totals([record])["unmeasured_runs"], 0)

    def test_eval_retries_continue_own_candidate(self):
        calls = []

        def attempt(task, kb, model, feedback, backend, starting_code):
            calls.append((kb is not None, starting_code, feedback))
            success = starting_code == FIXED
            return {"fixed_code": FIXED, "passed": success, "grade": PASS if success else FAIL,
                    "seconds": 0, "critic": {"feedback": "must never enter B feedback"}}

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run, "load", return_value=[TASK]), \
                patch.object(run, "KnowledgeBase"), patch.object(run, "one", side_effect=attempt), \
                patch.object(run, "show_attempt"), patch.object(run, "print_total"), patch.object(run, "console"):
            run.evaluate(Path(tmp), "gpt-5.5", "sdk")
        self.assertEqual([(kb, code) for kb, code, _ in calls],
                         [(False, None), (False, FIXED), (True, None), (True, FIXED)])
        self.assertTrue(all("must never" not in feedback for _, _, feedback in calls))

    def test_deep_tools_and_stop_on_pass_without_api(self):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_openai import ChatOpenAI

        visible_tools = []
        report = agent.RepairReport(intended_behavior="example", parameters_to_check="example",
                                    root_cause="example", bug_line=1, diagnostic_cue="example",
                                    evidence="example", fix_summary="example")

        def respond(model, messages, **kwargs):
            names = {t.get("function", t).get("name") for t in kwargs.get("tools", [])}
            visible_tools.append(names)
            calls = []
            if len(visible_tools) == 1:
                calls = [{"name": "replace_source_line", "args": {"line_number": 1, "new_line": FIXED.strip()}, "id": "edit"}]
            elif len(visible_tools) == 2:
                calls = [{"name": "simulate_fix", "args": {}, "id": "simulate"}]
            message = AIMessage(content="" if calls else report.model_dump_json(), tool_calls=calls,
                                usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                                response_metadata={"model_name": "gpt-5.5"})
            return ChatResult(generations=[ChatGeneration(message=message)])

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), \
                patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=respond), \
                patch("source_workspace.grade", return_value=PASS):
            answer, trace = deep_agent.repair_deep(TASK, [], "", "gpt-5.5")
        expected = {"search_source", "read_source_lines", "replace_source_line", "apply_source_patch", "simulate_fix"}
        self.assertEqual(visible_tools, [expected, expected, set()])
        self.assertEqual(answer.fixed_code, FIXED)
        self.assertEqual(trace["requests"], 3)
        self.assertEqual(trace["input_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
