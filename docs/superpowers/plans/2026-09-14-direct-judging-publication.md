# Direct Judging Publication Preparation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare, but do not push, the direct-rating parser repair and the available science direct-judging logs.

**Architecture:** Keep production code in the parent `methodology` branch and evaluation logs in the separate `LASR-G5/judging` repository. Keep current judging data under `judge/`, archive the other top-level data directories under `legacy_2/`, and publish direct logs under `direct/science/<semester>/<judge>/<condition>/` without copying the surrounding artifact tree.

**Tech Stack:** Python, pytest, Git, Git LFS, Inspect AI `.eval` archives.

**Spec:** `HANDOFF-direct-judging.md` and the user's Hugging Face Git workflow in this conversation.

## Global Constraints

- Do not push either repository without explicit user approval.
- Preserve unrelated worktree changes.
- Use Git only for Hugging Face publication.
- Exclude recovery keys and non-`.eval` campaign artifacts.
- HF default branch is `main`.

---

### Task 1: Prepare parser repair

**Files:**
- Modify: `src/judge/scorer.py`
- Test: `tests/test_direct_judge.py`

**Interfaces:**
- Consumes: raw direct-judge response text.
- Produces: `parse_direct_judge_output()` ratings on the inclusive 1–10 scale.

- [x] Review the existing repair against retained invalid outputs.
- [x] Run `pytest tests/test_direct_judge.py -q`.
- [x] Commit only `src/judge/scorer.py` and `tests/test_direct_judge.py` locally.
- [x] Incorporate `origin/methodology` without touching unrelated changes.

### Task 2: Prepare science log publication

**Files:**
- Move: `logs/{aime,apex,invalid,pilot,pilot_reasoning,rank,smoke}`
- Into: `logs/legacy_2/`
- Copy: `artifacts/direct-judging-science/logs/direct/science/**/*.eval`
- Into: `logs/direct/science/**/*.eval`

**Interfaces:**
- Consumes: completed Inspect AI `.eval` archives.
- Produces: a local commit on the judging repository's `main` branch.

- [x] Check out `LASR-G5/judging` at `main` in ignored `logs/`.
- [x] Keep `logs/judge/` and `logs/legacy/` unchanged; move every other top-level data directory to `logs/legacy_2/`.
- [x] Validate each source archive with Inspect AI and inventory successful, cancelled, and failed statuses.
- [x] Copy only `.eval` files, preserving the semester/judge/condition hierarchy.
- [x] Confirm no private keys or unrelated artifacts are staged.
- [x] Commit locally without pushing.

### Task 3: Verify prepared state

**Files:**
- Inspect: parent and judging Git status/logs.

**Interfaces:**
- Consumes: the two local commits.
- Produces: exact commit IDs and commands awaiting approval.

- [x] Verify focused tests pass.
- [x] Verify the judging commit contains only intended science `.eval` paths.
- [x] Verify neither local commit exists on its remote.
- [x] Report commit IDs and wait for explicit push approval.
