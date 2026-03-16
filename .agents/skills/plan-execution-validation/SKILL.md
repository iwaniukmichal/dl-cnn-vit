---
name: plan-execution-validation
description: Read a project plan from references, find the end-to-end experiment instructions in the README, and verify that the README instructions fully cover the project plan — fixing README or implementation gaps as needed.
---

# Plan Execution Validation

## Goal

Verify that the project plan can be **fully executed** by following the instructions in the repository README. The skill performs a structured cross-reference between:

1. the **project plan** (the source of truth for what must be done), and
2. the **README end-to-end experiment instructions** (the intended step-by-step guide for fulfilling the plan).

When gaps are found:
- if instructions are missing or unclear → update the README,
- if the implementation does not support a required plan element → update the system code/configs,
- if both are needed → fix both.

The **main deliverable** is a validated README whose end-to-end instructions, when followed exactly, produce all results the project plan demands.


## Required inputs

The repository should contain or provide:

- a project plan file in `references/` or `docs/` (LaTeX, Markdown, PDF, or similar),
- a `README.md` with a section describing how to run end-to-end experiments (e.g. `## End-to-end experiments` or similar),
- the implemented codebase (source code, CLI entrypoints, configs),

If the project plan is not in `references/`, the user must specify its location.

## Source-of-truth priority

Use sources in this order:

1. explicit user instructions,
2. the **project plan** (defines what must be achieved),
3. the **README end-to-end experiments section** (defines how a user would execute it),
4. the **actual codebase** (configs, CLI entrypoints, modules).

The project plan is the ultimate authority on **what** the project must accomplish.
The README is the authority on **how** a user would accomplish it.
The codebase is the authority on **what is currently possible**.

## Core rules

1. **The project plan defines completeness.**
   Every experiment, phase, deliverable, metric, and analysis described in the project plan must be achievable by following the README instructions.

2. **The README must be self-contained and executable.**
   A reader who follows the README end-to-end section step by step — without reading any additional plans — must produce all required results.

3. **Do not add unnecessary experiments.**
   If the project plan does not require something, do not add it to the README.

4. **Prefer README fixes over implementation changes.**
   If the implementation already supports a plan requirement but the README fails to document it, fix the README only. Change implementation only when the codebase genuinely cannot support a plan requirement.

5. **Preserve existing correct instructions.**
   Do not rewrite README sections that are already correct and complete. Only modify or add what is needed.

6. **Be explicit about manual steps.**
   If the workflow requires manual intervention, the README must document exactly what to inspect, what to copy, and where to paste it etc.


## Execution workflow

### Phase 1: Read and understand the project plan

Read the project plan file from `references/` (or the location specified by the user).

Extract a structured list of:

- project phases / work packages / milestones
- experiments required per phase
- models / architectures to be evaluated
- datasets and data regimes (full data, reduced data, few-shot episodes)
- hyperparameter search requirements
- augmentation studies
- analysis tasks (embedding analysis, clustering, visualization)
- ensemble or combination evaluations
- evaluation metrics expected
- deliverables and expected outputs per phase
- any ordering constraints or dependencies between phases
- reproducibility requirements (seeds, aggregation)

Produce a **plan requirements checklist** — a flat list of concrete items that the project must accomplish.

### Phase 2: Read and parse the README experiments section

Read `README.md` and locate the end-to-end experiment section. 
If no such section exists, flag this as a critical gap.

From the README section, extract:

- numbered steps in intended execution order
- CLI commands with their config files and arguments
- manual actions described between steps (e.g. "inspect results and propagate best config")
- inter-phase dependencies
- aggregation / summary steps
- any notes or caveats

Produce a **README instruction inventory** — a structured list of what the README tells a user to do.

### Phase 3: Cross-reference plan requirements against README instructions

For each item in the plan requirements checklist, determine:

| Status                | Meaning                                                                                 |
| --------------------- | --------------------------------------------------------------------------------------- |
| ✅ Covered            | The README has a clear, executable instruction that fulfills this plan requirement      |
| ⚠️ Partially covered  | The README mentions it but the instruction is incomplete, ambiguous, or missing details |
| ❌ Missing            | The plan requires it but the README has no corresponding instruction                    |
| 🔧 Implementation gap | The plan requires it but the codebase cannot currently support it                       |

For partially covered items, specify exactly what is missing (e.g. "missing config file reference", "no seed specified", "no aggregation step").

### Phase 4: Validate against the actual codebase

For every CLI command in the README:

1. verify the CLI entrypoint module exists,
2. verify the referenced config file exists,
3. verify the config file's `mode` matches the CLI entrypoint,
4. verify the output paths are consistent with the documented output conventions.

For every manual action in the README:

1. verify that the referenced files/directories would exist at that point in the workflow,
2. verify the described action is possible (e.g. propagating hyperparameters from search results into downstream configs).

Flag any:

- referenced config files that do not exist,
- CLI entrypoints that do not exist,
- output paths that would not be created by the described commands,
- impossible or contradictory instructions.

### Phase 5: Produce a validation report

Present findings in this structure:

#### 5a. Plan requirements checklist with status

List every plan requirement and its coverage status (✅, ⚠️, ❌, or 🔧).

#### 5b. Gaps and issues

For each gap or issue, describe:

- what the plan requires,
- what the README currently says (or doesn't say),
- what the codebase supports (or doesn't),
- the recommended fix (README change, implementation change, or both),
- severity (critical = blocks plan completion, minor = unclear but executable, cosmetic = could be clearer).

#### 5c. Proposed changes

List the concrete changes to be made, grouped by:

- **README changes**: new steps, clarified instructions, added commands, fixed references
- **Implementation changes**: new configs, new CLI flags, new modules, bug fixes
- **Config changes**: new YAML files, corrected YAML fields

### Phase 6: Apply fixes

After presenting the validation report, apply the proposed changes:

1. **README fixes first.** Edit the README to add missing steps, fix commands, clarify manual actions, and correct references. Follow these sub-rules:
   - maintain the existing README style and formatting,
   - keep the same section hierarchy,
   - add new steps in the correct execution order,
   - preserve existing correct content verbatim,
   - ensure every new command references a real, existing config file.

2. **Implementation fixes second.** If the codebase lacks support for a plan requirement:
   - create missing config files following existing config conventions,
   - add missing CLI entrypoint capabilities if needed,
   - add minimal module changes only when strictly necessary,
   - do not refactor or restructure existing working code.

3. **Verify fixes.** After applying changes:
   - re-read the modified README and confirm every plan requirement now has a corresponding instruction,
   - confirm every referenced config/module exists,
   - if the repository has tests, run them to confirm nothing is broken.

## What to avoid

- Inventing experiments not in the project plan.
- Rewriting the entire README when only targeted additions are needed.
- Restructuring the codebase for aesthetic reasons.
- Adding orchestration or automation the plan does not require.
- Silently dropping plan requirements that are hard to validate.
- Guessing what the plan means when it is ambiguous — ask the user instead.
- Making the README longer or more complex than necessary.
- Removing existing correct README content.

## Required response pattern

When using this skill, structure the work around this pattern:

### 1. Project plan understanding

Summarize the project plan's phases, experiments, and deliverables.

### 2. README instruction inventory

Summarize what the README currently instructs a user to do.

### 3. Cross-reference validation

Present the plan requirements checklist with coverage status for each item.

### 4. Gaps and issues

Detail each gap or issue found, with recommended fix and severity.

### 5. Proposed changes

List planned README, implementation, and config changes.

### 6. Applied changes

Describe every change made, with file paths and brief rationale.

### 7. Final validation state

State whether the README now fully covers the project plan, and list any remaining blockers or caveats.

## Edge cases

### If the README has no end-to-end section

Create one from scratch following the project plan phases, using the existing codebase and config files to construct the exact commands.

### If the project plan is ambiguous

List the ambiguities explicitly and ask the user for clarification before making changes. Do not guess.

### If the implementation has a real gap

Describe what is missing, propose the minimal implementation change, and implement it if the change is small and well-scoped. For larger changes, describe the change and ask the user for approval before proceeding.

### If the README has extra steps not in the plan

Leave them in place unless they contradict the plan. Note them as "extra, not required by the plan" in the validation report.

### If multiple valid execution orders exist

Document the one that most closely matches the project plan's phase ordering. Note alternatives only if they matter for correctness.
