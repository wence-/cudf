# Evaluation report

On September 28, 2026, the skill was evaluated on four tasks with and without the skill in Claude Code and Codex. A cleanup failure in the original Claude trials led to an explicit exception-cleanup requirement in the main workflow. Fresh targeted trials of the revised skill passed the independent worker-handoff checks in both harnesses. The original comparison and subsequent regression checks are reported separately below.

## Original task results

One fresh agent trial was run per task, harness and condition, for 16 trials. The native tasks used synthetic C++ starters and an independent checker maintained outside this distribution. The checker preserved the supplied caller and CPU implementation. The native test suite is not included, so these GPU results cannot be reproduced from this package alone.

| Task | Claude without skill | Claude with skill | Codex without skill | Codex with skill |
| --- | --- | --- | --- | --- |
| Native filter, multiplication and aggregation | Pass | Pass | Pass | Pass |
| Native worker handoff, including allocation-failure cleanup | Fail | Fail | Pass | Pass |
| Ask for missing engine inputs without inventing a build result | Pass | Pass | Pass | Pass |
| Return pandas expression retaining missing group keys | Pass | Pass | Pass | Pass |

For worker handoff, the second output allocation was made after a device-to-host copy had been queued. Both Claude implementations released the first pinned output buffer during exception unwinding before its copy completed. The independent observer reported early_release=1. Both agents had reported successful tests, so the independent result overrides their final summaries. The observer waited before actually freeing the buffer, preventing the test from turning the detected lifetime violation into a use-after-free.

Codex passed the normal, empty, boundary, delayed-upload, allocation-failure and subsequent-call checks in both conditions. Filter/aggregate comparisons use exact sum and count equality. Worker comparisons use exact ordered values and doubles, plus ownership/event observations. These checks establish the observed cases only; they are not a proof of every exception path.

## Cleanup correction and regression checks

The main workflow now requires inspecting operations that can throw after GPU submission, preserving owners through cleanup, testing a recoverable allocation failure while work is pending, and checking a later valid call when the engine promises recovery. A failed ownership or readiness check is a failed POC even if normal output values match.

The worker-handoff task was then rerun from the original starter with the revised skill, once in each harness. Neither trial saw the earlier generated implementation or its failure report. Both generated implementations allocated both host output buffers before submitting either device-to-host copy, removing the observed second-allocation failure window. The independent checker passed normal results, empty and boundary inputs, delayed upload, both allocation failures and subsequent valid calls. Nonempty CUDA traces and source inspection confirmed result-producing libcudf work.

| Revised-skill trial | Independent checks | Input tokens including cache | Output tokens | Elapsed seconds |
| --- | --- | ---: | ---: | ---: |
| Claude Code / claude-sonnet-5 | Pass | 238,004 | 21,121 | 225.7 |
| Codex / gpt-5.5, medium reasoning | Pass | 519,397 | 6,603 | 165.1 |

These two regression trials do not replace the original failures and are not a new paired comparison against the baseline. Other runtime failures, including lost-device recovery and every possible throwing library call, remain outside the measured coverage.

## Skill selection

Both harnesses read SKILL.md and its asynchronous-cleanup reference for both native tasks in the with-skill condition. Neither loaded the skill for the pandas-only negative prompt. For the missing-input prompt, Codex read the skill; Claude asked appropriate prerequisite questions without reading it. Claude therefore met the behavioral expectation but not the dataset's expected skill activation for that prompt.

The harness exposed the skill name, description and readable file path only in the with-skill condition. Claude ran with customizations disabled to isolate that choice; Codex used isolated runtime state. These are controlled selection observations, not a test of discovery after installing from the public catalog.

## Token use and elapsed time

The table totals all four trials in each condition. Input totals include cached input. Output totals are the harness-reported output counts. The harnesses report usage differently, so compare conditions within a harness, not token totals across models.

| Harness and model | Condition | Input tokens including cache | Output tokens | Total elapsed seconds |
| --- | --- | ---: | ---: | ---: |
| Claude Code 2.1.280 / claude-sonnet-5 | without | 583,799 | 24,194 | 674.7 |
| Claude Code 2.1.280 / claude-sonnet-5 | with | 469,027 | 30,743 | 323.2 |
| Codex CLI 0.155.1 / gpt-5.5, medium reasoning | without | 1,197,025 | 11,611 | 324.4 |
| Codex CLI 0.155.1 / gpt-5.5, medium reasoning | with | 1,251,820 | 11,759 | 314.7 |

Claude ran on Windows and used a restricted helper to build and execute on the Linux GPU host. Codex ran inside the Linux container. GPU work was serialized; Claude elapsed time includes waiting for other tests. These times are not comparable query-speed measurements or evidence of generation-speed improvement.

## Environment and method

Native builds used one NVIDIA L40S, CUDA toolkit 12.9.86, libcudf 26.08.01, RMM 26.08.0, GCC 13.3.0 and a C++20 build. Each container was limited to four CPUs and 16 GiB memory. Native tasks permitted up to four build attempts within a single trial. Setup failures before usable evaluation were excluded from the reported trials.

After generation, an independent checker rebuilt the saved programs and ran the synthetic fixtures, allocation failures and same-process recovery checks. CUDA kernel traces were collected separately for nonempty runs. Trace review and source review must establish result-producing libcudf work; numerical agreement alone does not establish GPU execution.

## CI and GPU evaluation split

evals/evals.json contains the distributed CPU-only checks for missing prerequisites and a pandas-only negative request. The native GPU suite and comparison checker are maintained separately and are not distributed here. The configuration does not provision a GPU. Passing the distributed checks does not establish native GPU correctness. See evals/EVAL.md for the available evaluation procedure.

## Evaluated revision and limits

Evaluated SKILL.md SHA256: 422fc5e85e25fb20e494ee2903820f33910509d866334785e918a265c5ca68b9.

The revised skill used for the cleanup regression has SKILL.md SHA256 0ab905f2a5274f78a3da999541a0f6e32ef1748cb56d2f19740f6e40cbe4cf35. It adds the exception-cleanup requirement. The full four-task comparison belongs to the original revision; the targeted regression belongs to this revised file.

This is one trial per task and condition on two native task families. It does not establish a general quality lift, a token saving, query speed, production readiness, multi-GPU behavior or distributed execution.
