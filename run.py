"""Train the RCA KB on set A, then compare frozen-KB and baseline on set B."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from agent import repair, usage_counts
from critic import review
from deep_agent import repair_deep
from grade import grade
from kb import KnowledgeBase
from scorecard import scorecard

ROOT = Path(__file__).resolve().parent
console = Console(highlight=False)
# Published standard API rates, USD per million input/cached-input/output tokens.
PRICES_PER_MILLION = {"gpt-5.5": (5.0, 0.5, 30.0)}
SCORER_VERSION = "semantic_cause_v4"


def estimated_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    rates = PRICES_PER_MILLION.get(model)
    if not rates:
        return None
    input_rate, cached_rate, output_rate = rates
    return round(((input_tokens - cached_tokens) * input_rate
                  + cached_tokens * cached_rate + output_tokens * output_rate) / 1_000_000, 6)


def show_attempt(set_name: str, index: int, total: int, task: dict,
                 record: dict, attempt: int | None = None, condition: str = "") -> None:
    card = record.get("scorecard", {})
    grade_result = record.get("grade", {})
    mismatch = grade_result.get("mismatches")
    samples = grade_result.get("samples")
    sim = f"{mismatch}/{samples}" if mismatch is not None else (
        "error" if grade_result else "not run")
    tokens = (f"{record['input_tokens']:,}/{record['output_tokens']:,}"
              if "input_tokens" in record else "n/a")
    short_id = task["id"].split("_", 1)[0]
    status = ("[green]SIM PASS[/]" if record["passed"] else
              "[red]SIM FAIL[/]" if grade_result else "[red]AGENT ERROR[/]")
    if "qualified" not in record:
        gate = "[dim]evaluation (no training gate)[/]"
    elif record["qualified"]:
        gate = "[green]QUALIFIED[/]"
    elif record["passed"]:
        gate = "[yellow]NOT QUALIFIED: score <= 0.80[/]"
    elif not grade_result:
        gate = "[red]NOT QUALIFIED: no final repair[/]"
    else:
        gate = "[red]NOT QUALIFIED: simulation[/]"
    label = f"try {attempt}" if attempt is not None else condition
    console.print(f"[bold]{set_name} {index:02d}/{total:02d} {short_id}[/]  {label:<8}  "
                  f"{status}  |  {gate}")
    score_text = (f"{card['score']:.2f} (code {card['code_similarity']:.2f}, "
                  f"cause {card['cause_overlap']:.2f}, line {card['bug_line_proximity']})") \
        if card else "n/a"
    console.print(f"    sim {sim}  |  score {score_text}")
    console.print(f"    tokens {tokens}  |  time {record['seconds']:.1f}s  |  "
                  f"cost ~{format_cost(record.get('estimated_model_cost_usd'))}  |  "
                  f"tools {record.get('tool_calls', 0)}")
    console.print(f"    KB run: {record.get('run_kb_action', 'no update (evaluation)')}  |  "
                  f"master: {record.get('master_kb_action', 'no update (evaluation)')}")
    if record.get("error") or record.get("critic_error"):
        console.print(f"    error: {record.get('error') or record['critic_error']}")


def totals(records: list[dict]) -> dict:
    costs = [row["estimated_model_cost_usd"] for row in records
             if row.get("estimated_model_cost_usd") is not None]
    return {
        "input_tokens": sum(row.get("input_tokens", 0) for row in records),
        "output_tokens": sum(row.get("output_tokens", 0) for row in records),
        "seconds": round(sum(row["seconds"] for row in records), 3),
        "estimated_model_cost_usd": round(sum(costs), 6) if costs else None,
        "unmeasured_runs": sum("input_tokens" not in row or row.get("usage_incomplete", False)
                               for row in records),
    }


def average_scores(records: list[dict]) -> dict:
    # Failed/unscored tasks contribute zero; both conditions use all tasks.
    return {key: round(sum(r.get("scorecard", {}).get(key, 0.0) for r in records) / len(records), 3)
            if records else 0.0
            for key in ("score", "code_similarity", "cause_overlap", "bug_line_proximity")}


def format_average(card: dict) -> str:
    return (f"score {card['score']:.2f} (code {card['code_similarity']:.2f}, "
            f"cause {card['cause_overlap']:.2f}, line {card['bug_line_proximity']:.2f})")


def format_cost(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "n/a"


def print_total(label: str, row: dict) -> None:
    console.rule(f"[bold]{label}[/]")
    outcome = "Qualified" if label == "SET A TOTAL" else "Simulator passed"
    console.print(f"[bold]{outcome}: {row['passed']}/{row['total']}[/]   "
                  f"{format_average(row['scorecard_average'])}")
    console.print(f"Score coverage: {row['scored_tasks']}/{row['total']} (unscored tasks count as zero)")
    if "first_attempt_passed" in row:
        console.print(f"First attempt: {row['first_attempt_passed']}/{row['total']}   "
                      f"Attempts used: {row['attempts_used']}")
    console.print(f"Tokens: {row['input_tokens']:,} in / {row['output_tokens']:,} out   "
                  f"Time: {row['seconds']:.1f}s")
    console.print(f"Estimated model cost: {format_cost(row['estimated_model_cost_usd'])}   "
                  f"Unmeasured runs: {row['unmeasured_runs']}")


def load(name: str) -> list[dict]:
    return json.loads((ROOT / "data" / f"{name}.json").read_text(encoding="utf-8"))


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def one(task: dict, kb: KnowledgeBase | None, model: str, feedback: str = "", backend: str = "sdk",
        starting_code: str | None = None) -> dict:
    started = time.perf_counter()
    trace = {}
    try:
        hints = kb.hints(task["description"], task["category"]) if kb else []
        answer, trace = (repair_deep if backend == "deepagents" else repair)(task, hints, feedback, model, starting_code)
        # Reuse the agent's final simulation when its returned source is unchanged.
        result = trace.get("verified_grade") or grade(task, answer.fixed_code)
        record = {
            "task_id": task["id"], "category": task["category"],
            "passed": result["passed"], "grade": result,
            "root_cause": answer.root_cause, "fix_summary": answer.fix_summary,
            "intended_behavior": answer.intended_behavior,
            "parameters_to_check": answer.parameters_to_check,
            "diagnostic_cue": answer.diagnostic_cue, "evidence": answer.evidence,
            "bug_line": answer.bug_line,
            "fixed_code": answer.fixed_code, "kb_hits": [h["task_id"] for h in hints],
            "input_tokens": trace["input_tokens"], "output_tokens": trace["output_tokens"],
            "cached_input_tokens": trace.get("cached_input_tokens", 0),
            "estimated_model_cost_usd": estimated_cost(
                model, trace["input_tokens"], trace.get("cached_input_tokens", 0), trace["output_tokens"]),
            "requests": trace["requests"], "simulations": trace.get("simulations", []),
            "tool_calls": trace.get("tool_calls", 0), "tool_history": trace.get("tool_history", []),
            "usage_incomplete": trace.get("usage_incomplete", False),
            "prompt": trace["prompt"],
        }
        try:
            critique, usage = review(task, record, model)
            record["critic"] = critique.model_dump()
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "requests"):
                record[key] += usage[key]
            record["estimated_model_cost_usd"] = estimated_cost(
                model, record["input_tokens"], record["cached_input_tokens"], record["output_tokens"])
            record["scorecard"] = scorecard(task, answer.fixed_code, critique.cause_score, answer.bug_line)
        except Exception as exc:
            record["critic_error"] = f"{type(exc).__name__}: {exc}"
            data = getattr(exc, "run_data", None)
            if data is not None:
                for key, value in usage_counts(data.context_wrapper.usage).items():
                    record[key] += value
                record["estimated_model_cost_usd"] = estimated_cost(
                    model, record["input_tokens"], record["cached_input_tokens"], record["output_tokens"])
            record["usage_incomplete"] = True
            # A failed judge is unscored, not evidence of an incorrect root cause.
        record["seconds"] = round(time.perf_counter() - started, 3)
        return record
    except Exception as exc:
        trace = getattr(exc, "trace", trace)
        record = {**trace, "task_id": task["id"], "category": task["category"],
                  "passed": False, "error": f"{type(exc).__name__}: {exc}",
                  "usage_incomplete": trace.get("usage_incomplete", True),
                  "seconds": round(time.perf_counter() - started, 3)}
        if trace.get("verified_grade") is not None:
            record["grade"] = trace["verified_grade"]
            record["passed"] = record["grade"]["passed"]
        if "input_tokens" in trace:
            record["estimated_model_cost_usd"] = estimated_cost(
                model, trace["input_tokens"], trace.get("cached_input_tokens", 0), trace["output_tokens"])
        return record


def train(run_dir: Path, model: str, attempts: int, backend: str) -> bool:
    # Run KB starts empty; successful lessons also update the persistent master.
    kb = KnowledgeBase(run_dir / "kb")
    master = KnowledgeBase(ROOT / "results" / "master_kb")
    tasks = load("set_a")
    console.rule("[bold]Train Set A[/]")
    console.print(f"Tasks: {len(tasks)}   Data: data/set_a.json")
    console.print("Gate: simulation pass AND score > 0.80. Provisional lessons update only the run KB.")
    console.print(f"Run KB: {run_dir / 'kb'}   Master entries: {master.count()}")
    records = []
    for index, task in enumerate(tasks, 1):
        feedback = ""
        candidate = None
        best_quality = (-1, -1, -1.0)
        for attempt in range(1, attempts + 1):
            record = one(task, kb, model, feedback, backend, candidate)
            candidate = record.get("fixed_code", candidate)
            record["attempt"] = attempt
            score = record.get("scorecard", {}).get("score", 0.0)
            record["qualified"] = record["passed"] and score > 0.8

            # Critic feedback guides only set A retries. Set B is scored with
            # the same grader, but its feedback never reaches the fixer or KB.
            critique = record.get("critic")

            # Prefer qualified, then simulator-passing, then higher-scoring
            # lessons. A passing repair must outrank a failed simulation.
            quality = (int(record["qualified"]), int(record["passed"]), score)
            record["run_kb_action"] = "skipped (no better lesson)"
            record["master_kb_action"] = "skipped (not qualified)"
            if quality > best_quality and (record["qualified"] or critique is not None):
                kb.save(task, record if record["qualified"] else critique)
                record["run_kb_action"] = ("updated (verified)" if record["qualified"]
                                           else "updated (provisional)")
                best_quality = quality
            elif critique is None:
                record["run_kb_action"] = "skipped (no lesson)"
            if record["qualified"]:
                master.save(task, record)
                record["master_kb_action"] = "updated (verified)"

            records.append(record)
            save(run_dir / "set_a_trajectories.json", records)
            show_attempt("A", index, len(tasks), task, record, attempt=attempt)
            if record["qualified"]:
                break
            feedback = (critique["feedback"] if critique else
                        f"Simulator result: {record.get('grade', record.get('error'))}.")
    latest = {task["id"]: next((r for r in reversed(records) if r["task_id"] == task["id"]), None)
              for task in tasks}
    passed = sum(bool(r and r["qualified"]) for r in latest.values())
    summary = {"passed": passed, "total": len(latest), "pass_rate": passed / len(latest),
               "simulator_passed": sum(bool(r and r["passed"]) for r in latest.values()),
               "qualification_rule": "sim_pass_and_score_gt_0.8",
               "scorer_version": SCORER_VERSION,
               "kb_entries": kb.count(), "master_kb_entries": master.count(),
               "model": model, "backend": backend, "attempt_limit": attempts,
               "evaluation_unlocked": passed == len(latest), **totals(records),
               "scored_tasks": sum(bool(r and r.get("scorecard")) for r in latest.values()),
               "scorecard_average": average_scores([r for r in latest.values() if r])}
    lessons = kb.export()
    save(run_dir / "kb_lessons.json", lessons)
    save(run_dir / "set_a_summary.json", summary)
    print_total("SET A TOTAL", summary)
    return summary["evaluation_unlocked"]


def evaluate(run_dir: Path, model: str, backend: str, mode: str = "both",
             max_attempts: int = 3) -> None:
    # Evaluation uses only the run KB and never upserts into it or the master.
    kb = KnowledgeBase(run_dir / "kb", read_only=True)
    name = ("set_b_comparison" if mode == "both" else f"set_b_{mode}") + f"_{max_attempts}tries"
    tasks = load("set_b")
    console.rule("[bold]Evaluate Set B[/]")
    console.print(f"Tasks: {len(tasks)}   Mode: {mode}   Data: data/set_b.json")
    console.print(f"Retry limit: {max_attempts} per condition; feedback uses simulator results only.")
    console.print(f"Frozen KB: {run_dir / 'kb'}   Entries: {kb.count()}")
    comparisons = []
    for index, task in enumerate(tasks, 1):
        pair = {"task_id": task["id"]}
        for label, active_kb in (("baseline", None), ("with_kb", kb)):
            if mode != "both" and mode != ("kb" if active_kb else "baseline"):
                continue
            attempts = []
            feedback = ""
            candidate = None
            for attempt in range(1, max_attempts + 1):
                record = one(task, active_kb, model, feedback, backend, candidate)
                candidate = record.get("fixed_code", candidate)
                record["attempt"] = attempt
                record["run_kb_action"] = ("frozen (read only)" if active_kb
                                           else "not used (baseline)")
                record["master_kb_action"] = "not updated (evaluation)"
                attempts.append(record)
                show_attempt("B", index, len(tasks), task, record,
                             condition=f"{label} try {attempt}")
                if record["passed"]:
                    break
                feedback = f"Previous simulator failure: {record.get('grade', record.get('error'))}."
            pair[label] = attempts[-1]
            pair[f"{label}_attempts"] = attempts
        comparisons.append(pair)
        save(run_dir / f"{name}.json", comparisons)
    summary = {}
    for label in ("baseline", "with_kb"):
        rows = [pair[label] for pair in comparisons if label in pair]
        if not rows:
            continue
        all_attempts = [attempt for pair in comparisons if label in pair
                        for attempt in pair[f"{label}_attempts"]]
        summary[label] = {"passed": sum(row["passed"] for row in rows),
                          "first_attempt_passed": sum(pair[f"{label}_attempts"][0]["passed"]
                                                      for pair in comparisons if label in pair),
                          "attempts_used": len(all_attempts), "attempt_limit": max_attempts,
                          "total": len(rows), **totals(all_attempts),
                          "scored_tasks": sum(bool(row.get("scorecard")) for row in rows),
                          "scorecard_average": average_scores(rows)}
    save(run_dir / f"{name}_summary.json", summary)
    for label, row in summary.items():
        print_total(f"SET B {label}", row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--eval-mode", choices=["both", "kb", "baseline"], default="both")
    parser.add_argument("--model")
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--eval-attempts", type=int, default=3)
    parser.add_argument("--backend", choices=["sdk", "deepagents"])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.attempts < 1 or args.eval_attempts < 1:
        parser.error("Attempt limits must be at least 1")
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required in the process environment")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or ROOT / "results" / stamp
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if args.phase == "eval":
        summary_path = run_dir / "set_a_summary.json"
        if not summary_path.exists():
            parser.error(f"No completed set A training in {run_dir}")
        training = json.loads(summary_path.read_text(encoding="utf-8"))
        if training.get("qualification_rule") != "sim_pass_and_score_gt_0.8":
            parser.error("Training used an older pass rule; train a fresh run before evaluation")
        if training.get("scorer_version") != SCORER_VERSION:
            parser.error("Training used an older root-cause scorer; train a fresh run before evaluation")
        if not training["evaluation_unlocked"]:
            parser.error("Set A is below 100%; set B evaluation is locked")
        model = args.model or training["model"]
        backend = args.backend or training["backend"]
        if model != training["model"] or backend != training["backend"]:
            parser.error("Evaluation model and backend must match set A training")
        evaluate(run_dir, model, backend, args.eval_mode, args.eval_attempts)
        return
    if (run_dir / "set_a_summary.json").exists():
        parser.error(f"Training already exists in {run_dir}; choose another --run-dir")
    if (run_dir / "set_a_trajectories.json").exists() or (run_dir / "kb").exists():
        parser.error(f"Partial training exists in {run_dir}; choose a fresh --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = args.model or "gpt-5.5"
    backend = args.backend or "sdk"
    save(run_dir / "config.json", {"model": model, "attempts": args.attempts,
                                   "backend": backend,
                                   "set_a": "data/set_a.json", "set_b": "data/set_b.json"})
    unlocked = train(run_dir, model, args.attempts, backend)
    if args.phase == "all":
        if unlocked:
            evaluate(run_dir, model, backend, args.eval_mode, args.eval_attempts)
        else:
            print("Set A is below 100%; Set B evaluation was not run.", flush=True)


if __name__ == "__main__":
    main()
