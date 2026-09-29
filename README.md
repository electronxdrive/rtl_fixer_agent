# RTL Fixer Agent

A small RCA knowledge-base experiment on 20 disjoint ChipBench zero-shot Verilog debugging tasks. Set A contains 10 training cases; set B contains 10 held-out cases. The same Icarus Verilog grader checks every repair against ChipBench's supplied testbench.

## How it works

1. The OpenAI Agents SDK repair agent receives the specification and up to three same-category lessons from ChromaDB. It searches `dut.sv` with ripgrep, reads numbered snippets of at most 40 lines, edits a numbered line or applies a Git unified patch, and simulates the current file with Icarus Verilog. Both backends enforce 12 source-tool calls, at most 3 simulation calls, and a 14-model-request ceiling per attempt. Tools stop after simulation passes or their budget is exhausted. The optional Deep Agents backend exposes only the same five RTL tools; extra filesystem and delegation tools are disabled. Unchanged candidates reuse their simulation result.
2. Training retries each set A task up to six times, carrying the previous candidate into the next attempt. A reference-aware critic sees snippets around every changed region using each file's line coordinates, grades the attempted root cause by meaning, and gives the fixer concise feedback for the next attempt. The primary bug line can identify any real defect. Each repair records intended behavior, parameters to inspect, a diagnostic cue, root cause, evidence, and why the fix works. The KB stores this reviewable reasoning summary and its verification state. A better lesson updates the run KB; a repair updates the persistent master KB at `results/master_kb` only after it passes simulation **and** scores above 0.8.
3. Set B runs only if all 10 set A tasks meet both conditions. The run KB is read-only for set B. Each task runs without KB and with KB under the same retry limit (default three attempts per condition). A failed attempt passes only simulator feedback to the next attempt. Both conditions receive the same post-repair root-cause grading; critic feedback is never supplied to the set B fixer. Reports show first-attempt pass rate and pass rate after retries, with token, time, and cost totals across every attempt.
4. Stdout and JSON results report simulation and training qualification separately. Set A qualifies only when simulation passes **and** score is strictly above 0.80. Each attempt also states whether the run KB received a provisional or verified lesson, and whether the master KB was updated or skipped. A provisional run-KB lesson is replaced by a better one, preferring simulator-passing repairs before score. The scorecard shows code similarity, semantic root-cause correctness, and bug-line proximity, alongside tokens, time, and estimated cost. Set A's 100% target is a supervised training gate; it is **not** a held-out success rate.

## Run

Requirements: Python 3.12, `OPENAI_API_KEY`, and Icarus Verilog (`iverilog.exe` and `vvp.exe`). On this Windows machine the tools are at `C:\iverilog\bin`; change `grade.py` if installed elsewhere.

```powershell
cd C:\Users\srira\Documents\chip_agents\rtlx_fixer_agent
& ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')
& ..\.venv\Scripts\python.exe build_dataset.py
make train
make eval
# Or evaluate only with the frozen KB:
make eval KB=true
# Change the equal retry budget for both evaluation conditions:
make eval EVAL_ATTEMPTS=4
```

`make train` prints the training-data path, each set A attempt, and a set-level summary in `results/compact_v4/set_a_summary.json`. It must reach 10/10 qualified repairs before `make eval` runs. `make eval` prints each set B baseline/KB result and aggregate statistics. `make eval KB=true` runs only the KB condition; `make eval KB=false` runs only the baseline. To try the alternative Deep Agents harness, use `make train BACKEND=deepagents RUN_DIR=results/deep_run`, then `make eval RUN_DIR=results/deep_run`. Use a fresh `RUN_DIR` for a new training experiment. Each run starts with an empty run KB, while the master KB retains verified lessons across runs. The v4 scorer requires fresh training; existing experiment files are preserved.

## Data and grading

The selected `data/set_a.json` and `data/set_b.json` files contain ChipBench prompts, buggy RTL, reference RTL, and testbenches. Repair-agent prompts never include reference RTL. The reference source is used by the simulator grader and the post-repair critic in both sets. The grader requires zero reported mismatches and at least one tested sample. All 20 selected references passed their supplied tests in this environment.

The three scorecard values are diagnostics from the reference source and patch, not the pass criterion. Estimated model cost uses [published GPT-5.5 standard API rates](https://developers.openai.com/api/docs/models/gpt-5.5) and reported cached tokens. Embedding charges from Chroma retrieval are excluded. Runs with an API error may have unmeasured tokens and cost; stdout reports the count of such runs.

Score averages use every task, counting unscored tasks as zero and displaying score coverage. Interrupted agents retain their candidate, tool history, current simulation result, and available SDK/callback usage. Missing API usage remains explicitly marked incomplete. Set B retries carry only their own condition's candidate and simulator feedback; they never receive reference-aware critic feedback.

Source: [ChipBench](https://github.com/zhongkaiyu/ChipBench), MIT license copied to `data/CHIPBENCH_LICENSE.txt`. The HWE-Bench source snapshot is available beside this project but is not used in this experiment.
