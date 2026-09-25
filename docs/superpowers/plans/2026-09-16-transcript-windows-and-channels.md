# Transcript windows and channels: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Judge the frozen 2026-I coding and cyber train items pairwise under partial views of each transcript (a fraction of it by position, or one of its channels) so the results say where in a transcript the capability signal sits.

**Architecture:** Two new `JudgeConfig` fields, `window` and `channels`, ride the existing pairwise path: the renderer in `src/evals/sample.py` filters channels, `src/judge/dataset.py` cuts the window before the character limit, and the config's hash and log-directory label carry the condition. No new harness; collection is `uv run evaluate ... --judge <config>` as for the pairwise truncation runs.

**Tech Stack:** Python 3.12, inspect_ai, draccus configs, pytest via `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-09-16-transcript-windows-and-channels-design.md`

## Global constraints

- Work on branch `rq_analyses` in the main checkout. Run everything with `uv run`.
- Every judging run uses the frozen `2026-I_train` items (AGENTS.md rule); the config's `train_per_quartile` stays 0.
- Existing condition hashes and log directories must not change: new fields enter the hash only when set, and the full rendering is byte-identical when `channels` is empty.
- Names say what a thing is: `window-keep-2of10`, `channels-only-tools`. No opaque codes.
- Commit messages: `<area>: <what>`, lowercase, no attribution trailers.
- Paid collection (Task 6) launches only after the user has approved the condition set in that task.
- Never `git lfs pull` broadly in `data/` or `logs/`; the launch task says exactly which benches to pull.

---

### Task 1: config fields, hash and label

**Files:**
- Modify: `src/configs/judge.py` (fields at lines 19-22, `__post_init__` 34-38, `condition_hash` 40-48, `truncation_label` 50-53, `DirectJudgeConfig.condition_hash` 62-72)
- Modify: `src/protocols/pairwise/judge.py:78-82` (`log_dir`)
- Modify: `src/protocols/direct_judge/judge.py:70-80` (`log_dir`)
- Test: `tests/test_protocols.py`, `tests/test_direct_judge.py`

**Interfaces:**
- Produces: `JudgeConfig.window: str` (`""`, `keep-<k>/<n>` or `drop-<k>/<n>`), `JudgeConfig.channels: str` (`""`, `only-<c>` or `drop-<c>` with c in reasoning, text, tools), `JudgeConfig.condition_label: str` (replaces `truncation_label`), module-level `parse_window(spec) -> tuple[str, int, int]` and `CHANNELS = ("reasoning", "text", "tools")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_protocols.py`:

```python
def test_window_and_channels_are_conditions():
    cfg = {"name": "j", "judge": {"name": "p/j", "slug": "j"}}
    base = JudgeConfig.from_cli(cfg)
    assert base.condition_label == ""
    windowed = JudgeConfig.from_cli(cfg | {"window": "keep-2/10"})
    assert windowed.condition_hash != base.condition_hash and windowed.condition_label == "window-keep-2of10"
    channel = JudgeConfig.from_cli(cfg | {"channels": "only-tools"})
    assert channel.condition_hash not in (base.condition_hash, windowed.condition_hash)
    assert channel.condition_label == "channels-only-tools"
    both = JudgeConfig.from_cli(cfg | {"max_length": 2000, "truncate": "tail", "window": "drop-1/4"})
    assert both.condition_label == "truncated-tail-2000_window-drop-1of4"
    for bad in ("keep-2", "keep-0/4", "keep-5/4", "middle-1/4"):
        with pytest.raises(ValueError):
            JudgeConfig.from_cli(cfg | {"window": bad})
    for bad in ("tools", "only-images", "keep-tools"):
        with pytest.raises(ValueError):
            JudgeConfig.from_cli(cfg | {"channels": bad})
```

Add `import pytest` at the top of `tests/test_protocols.py` if missing. In `tests/test_direct_judge.py:36`, extend the loop of hash-changing edits with `{"window": "keep-1/4"}` and `{"channels": "drop-text"}`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_protocols.py::test_window_and_channels_are_conditions tests/test_direct_judge.py::test_direct_config_defaults_and_condition_identity`
Expected: FAIL (`unexpected keyword 'window'` / `AttributeError: condition_label`).

- [ ] **Step 3: Implement**

In `src/configs/judge.py`, after `FULL_LENGTH`:

```python
CHANNELS = ("reasoning", "text", "tools")  # what a rendered transcript is made of: CoT blocks, assistant prose, tool calls with their outputs


def parse_window(spec: str) -> tuple[str, int, int]:
    """("keep"|"drop", k, n) of a window spec "keep-k/n" / "drop-k/n": the k-th n-th of a transcript by characters."""
    mode, _, fraction = spec.partition("-")
    k, _, n = fraction.partition("/")
    if mode not in ("keep", "drop") or not (k.isdigit() and n.isdigit()) or not 1 <= int(k) <= int(n):
        raise ValueError(f"window must be keep-k/n or drop-k/n with 1 <= k <= n, got {spec!r}")
    return mode, int(k), int(n)


def parse_channels(spec: str) -> frozenset[str]:
    """The channels a spec shows: all of CHANNELS for "", one for "only-<c>", all but one for "drop-<c>"."""
    if not spec:
        return frozenset(CHANNELS)
    mode, _, channel = spec.partition("-")
    if mode not in ("only", "drop") or channel not in CHANNELS:
        raise ValueError(f"channels must be only-<c> or drop-<c> with c in {CHANNELS}, got {spec!r}")
    return frozenset([channel]) if mode == "only" else frozenset(CHANNELS) - {channel}
```

Add the fields after `truncate`:

```python
    window: str = ""  # a fraction of each transcript by characters, keep-k/n (show only the k-th n-th) or drop-k/n (hide it); applied before max_length
    channels: str = ""  # only-<c> shows one channel of the transcript, drop-<c> hides one: reasoning, text or tools (calls with their outputs)
```

In `__post_init__`, after the truncate check:

```python
        if self.window:
            parse_window(self.window)
        parse_channels(self.channels)
```

In both `condition_hash` properties, after the `truncate` clause:

```python
        if self.window:
            condition += ("window", self.window)
        if self.channels:
            condition += ("channels", self.channels)
```

Replace `truncation_label` with:

```python
    @property
    def condition_label(self) -> str:
        """Readable log-directory suffix of a partial-transcript condition, e.g. truncated-head-4000,
        window-keep-2of10, channels-only-tools (joined with "_" when combined); empty for full transcripts."""
        parts = []
        if self.max_length < FULL_LENGTH:
            parts.append(f"truncated-{self.truncate}-{self.max_length}")
        if self.window:
            parts.append(f"window-{self.window.replace('/', 'of')}")
        if self.channels:
            parts.append(f"channels-{self.channels}")
        return "_".join(parts)
```

In `src/protocols/pairwise/judge.py:80-81` and `src/protocols/direct_judge/judge.py:77-78` replace `truncation_label` with `condition_label` (comments: `# e.g. logs/judge_window-keep-2of10`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/test_protocols.py tests/test_direct_judge.py`
Expected: all pass, including the existing hash-stability check (`"7ea04f2b"`).

- [ ] **Step 5: Commit**

```bash
git add src/configs/judge.py src/protocols/pairwise/judge.py src/protocols/direct_judge/judge.py tests/test_protocols.py tests/test_direct_judge.py
git commit -m "judge: window and channels conditions on JudgeConfig, condition_label replaces truncation_label"
```

---

### Task 2: channel filtering in the renderer

**Files:**
- Modify: `src/evals/sample.py` (`format_content` 76-101, `format_single_message` 104-135, `format_message` 138-151, `extract_transcript` 154-180)
- Modify: `src/evals/transcripts.py` (`read_transcripts` 40-69, `_group` 83-100)
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `configs.judge.parse_channels`, `configs.judge.CHANNELS` (Task 1).
- Produces: `extract_transcript(sample, include_reasoning=True, channels="")`, `format_message(msg, include_reasoning=True, channels="")`, `read_transcripts(path, require_correct=True, include_reasoning=True, items=None, channels="")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sample.py` (its imports already cover the message and content classes and `ToolCall`):

```python
class TestChannels:
    """A channel spec keeps or drops one of reasoning, text and tools (calls with their outputs)."""

    turns = [
        ChatMessageAssistant(
            content=[ContentReasoning(type="reasoning", reasoning="Plan"), ContentText(type="text", text="Looking")],
            tool_calls=[ToolCall(id="1", function="bash", arguments={"cmd": "ls"})]),
        ChatMessageTool(content="file.txt", tool_call_id="1", function="bash"),
        ChatMessageAssistant(content=[ContentReasoning(type="reasoning", reasoning="Done"),
                                      ContentText(type="text", text="ANSWER: file.txt")]),
    ]

    def render(self, channels):
        return format_message(self.turns, channels=channels)

    def test_full_is_unchanged(self):
        assert self.render("") == format_message(self.turns)

    def test_only_one_channel(self):
        reasoning = self.render("only-reasoning")
        assert "Plan" in reasoning and "Done" in reasoning
        assert "Looking" not in reasoning and "bash" not in reasoning and "file.txt" not in reasoning
        text = self.render("only-text")
        assert "Looking" in text and "ANSWER" in text and "THOUGHT" not in text and "Tool" not in text
        tools = self.render("only-tools")
        assert "[Tool Call: bash" in tools and "[Tool Output (bash)]: file.txt" in tools
        assert "Plan" not in tools and "Looking" not in tools and "ANSWER" not in tools

    def test_drop_one_channel(self):
        assert "THOUGHT" not in self.render("drop-reasoning") and "Looking" in self.render("drop-reasoning")
        assert "Looking" not in self.render("drop-text") and "Plan" in self.render("drop-text")
        dropped = self.render("drop-tools")
        assert "Tool" not in dropped and "Plan" in dropped and "Looking" in dropped

    def test_empty_turns_vanish(self):
        """A turn with nothing left in it is not shown as an empty role line."""
        assert "[Tool]" not in self.render("only-text")
        assert not any(line.strip() == "[Assistant]:" for line in self.render("only-tools").splitlines())

    def test_reasoning_hidden_beats_channels(self):
        assert "THOUGHT" not in format_message(self.turns, include_reasoning=False, channels="only-reasoning")

    def test_bad_spec(self):
        import pytest
        with pytest.raises(ValueError):
            self.render("only-images")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_sample.py::TestChannels`
Expected: FAIL (`unexpected keyword argument 'channels'`).

- [ ] **Step 3: Implement**

In `src/evals/sample.py`, import `from configs.judge import CHANNELS, parse_channels` and:

```python
def format_content(content: Any, include_reasoning: bool = True, shown: frozenset[str] = frozenset(CHANNELS)) -> str:
    ...
    if isinstance(content, str):
        return format_text(content) if "text" in shown else ""
    ...
        if b_type == "text":
            if "text" in shown:
                text_val = getattr(block, "text", "")
                texts.append(format_text(str(text_val) if text_val is not None else ""))
        elif b_type == "reasoning":
            if include_reasoning and "reasoning" in shown and (rendered := format_reasoning(block)):
                texts.append(rendered)
        elif b_type in PLACEHOLDER_BLOCKS:
            if "text" in shown:
                texts.append(PLACEHOLDER_BLOCKS[b_type])
```

`format_single_message(msg, include_reasoning=True, shown=frozenset(CHANNELS))`: pass `shown` to `format_content`; return `""` for a tool-role message when `"tools" not in shown`; append the tool-call lines only when `"tools" in shown`.

`format_message(msg, include_reasoning=True, channels="")`: `shown = parse_channels(channels)`; pass `shown` down; in the list branch, when `channels` is non-empty skip turns whose `fmt` is `""` (the full rendering keeps them, so today's transcripts are unchanged).

`extract_transcript(sample, include_reasoning=True, channels="")`: pass `channels` to every `format_message` call.

In `src/evals/transcripts.py`: `read_transcripts(..., items=None, channels="")` adds `channels` to `key`; `_group(samples, require_correct, include_reasoning, channels="")` passes it to `extract_transcript`. Update the three `_group` calls.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/test_sample.py tests/test_dataset.py tests/test_historical.py`
Expected: all pass (the existing renderer tests prove the full rendering is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/evals/sample.py src/evals/transcripts.py tests/test_sample.py
git commit -m "evals: channel filter on the transcript renderer (reasoning, text, tools)"
```

---

### Task 3: window cut and plumbing through dataset, task and harnesses

**Files:**
- Modify: `src/judge/dataset.py` (`cut` 27-36, `responses_to_sample` 105-150, `build_dataset` 153-217, `build_direct_dataset` 220-241)
- Modify: `src/judge/task.py` (`judge_task` 18-88, `direct_judge_task` 91-116)
- Modify: `src/protocols/pairwise/judge.py:104-115`, `src/protocols/direct_judge/judge.py:95-104`
- Test: `tests/test_dataset.py`, `tests/test_task.py:47`, `tests/test_direct_judge.py`

**Interfaces:**
- Consumes: `configs.judge.parse_window` (Task 1), `read_transcripts(..., channels=)` (Task 2).
- Produces: `window_text(text, spec) -> str`; `responses_to_sample(..., truncate="head", window="")`; `build_dataset(..., truncate="head", window="", channels="")`; `build_direct_dataset(..., truncate="head", window="", channels="")`; `judge_task(..., truncate="head", window="", channels="")`; `direct_judge_task(..., truncate="head", window="", channels="")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset.py`:

```python
def test_window_modes():
    """keep-k/n shows the k-th n-th of the text by characters; drop-k/n elides it."""
    from judge.dataset import ENDS_SEPARATOR, window_text
    text = "".join(chr(ord("a") + i) for i in range(20))
    assert window_text(text, "") == text
    assert window_text(text, "keep-1/4") == text[:5] and window(text, "keep-4/4") == text[15:]
    assert window_text(text, "keep-3/10") == text[4:6]
    assert window_text(text, "drop-2/4") == text[:5] + ENDS_SEPARATOR + text[10:]
    assert window_text(text, "drop-1/4") == ENDS_SEPARATOR + text[5:]
    assert window_text(text, "drop-10/10") == text[:18] + ENDS_SEPARATOR
    assert "".join(window_text(text, f"keep-{k}/7") for k in range(1, 8)) == text  # windows tile the text
    with pytest.raises(ValueError):
        window_text(text, "keep-5/4")


def test_window_applies_before_cut_and_short_windows_are_skipped(monkeypatch):
    text = "".join(chr(ord("a") + i) for i in range(20))
    res = responses_to_sample(transcript(text, pad=False), transcript(text, pad=False), max_len=3, window="keep-2/4")
    assert res.metadata["chars_1"] == 3 and text[5:8] in res.input and text[:5] not in res.input
    # a window under MIN_RESPONSE_CHARS leaves nothing to judge: the pair is skipped like a short answer
    import judge.dataset as module
    long = "x" * 200
    monkeypatch.setattr(module, "read_transcripts",
                        lambda path, *a, **k: {"q1": [transcript(long, pad=False)]})
    monkeypatch.setattr(module, "read_header", lambda p: SimpleNamespace(eval=SimpleNamespace(model="m", dataset=SimpleNamespace(name="b"))))
    assert len(build_dataset(Path("a.eval"), Path("b.eval"), window="keep-1/4")) == 2
    assert len(build_dataset(Path("a.eval"), Path("b.eval"), window="keep-1/20")) == 0
```

In `tests/test_task.py:47` extend the expected `build_dataset` call with `truncate="head", window="", channels=""` (this also fixes the pre-existing failure). Add to `tests/test_direct_judge.py` next to the `shortened` case (line 76): a `build_direct_dataset(..., window="keep-1/2")` call asserting `sample.metadata["chars"]` is half the full length.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_dataset.py tests/test_task.py tests/test_direct_judge.py`
Expected: FAIL (`cannot import name 'window_text'`, unexpected keyword `window`).

- [ ] **Step 3: Implement**

In `src/judge/dataset.py`, import `parse_window` from `configs.judge` and add after `cut`:

```python
def window_text(text: str, spec: str) -> str:
    """The k-th n-th of the text by characters (keep-k/n), or the text without it and ENDS_SEPARATOR
    in its place (drop-k/n); "" leaves the text alone. Slices tile the text exactly."""
    if not spec:
        return text
    mode, k, n = parse_window(spec)
    start, stop = len(text) * (k - 1) // n, len(text) * k // n
    return text[start:stop] if mode == "keep" else text[:start] + ENDS_SEPARATOR + text[stop:]
```

`responses_to_sample(..., truncate="head", window="")`: `text_1 = cut(window_text(first.text, window), max_len, truncate)` and the same for `text_2`; the `truncated_*` metadata keeps comparing the raw length to `max_len`.

`build_dataset(..., truncate="head", window="", channels="")`: read transcripts with `channels=channels`; the `MIN_RESPONSE_CHARS` check and the `pair_kwargs` both include `window`. `build_direct_dataset` likewise: `text = cut(window_text(transcript.text, window), max_len, truncate)`.

`src/judge/task.py`: both task builders take `window: str = ""` and `channels: str = ""`, pass them to the dataset builders and record them in `metadata` next to `"truncate"`.

`src/protocols/pairwise/judge.py:110` and `src/protocols/direct_judge/judge.py:99`: pass `window=self.config.window, channels=self.config.channels`.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: 0 failures (the baseline had one, `test_judge_task_creation`, fixed here).

- [ ] **Step 5: Commit**

```bash
git add src/judge/dataset.py src/judge/task.py src/protocols/pairwise/judge.py src/protocols/direct_judge/judge.py tests/test_dataset.py tests/test_task.py tests/test_direct_judge.py
git commit -m "judge: window cut before the character limit; window and channels reach both judging paths"
```

---

### Task 4: judge configs for the condition set

**Files:**
- Create: `configs/judges/windowed/kimi-k2-thinking-{keep,drop}-<k>of<n>.yml` (keep 1..10 of 10, keep 1..4 of 4, drop 1..4 of 4: 18 files)
- Create: `configs/judges/channels/kimi-k2-thinking-{only,drop}-{reasoning,text,tools}.yml` (6 files)
- Create: `configs/judges/windowed/README.md`, `configs/judges/channels/README.md`
- Modify: `src/challenges/truncation/README.md` (one pointer line under "Pairwise truncation")

**Interfaces:**
- Consumes: `window` / `channels` config fields (Task 1). Each config is `configs/judges/reasoning/kimi-k2-thinking-full.yml` plus one field, like `configs/judges/truncated/kimi-k2-thinking-ends-2000.yml` is.

- [ ] **Step 1: Generate the files**

```bash
cd configs/judges && mkdir -p windowed channels
body='name: kimi-k2-thinking
judge: configs/models/moonshotai/kimi-k2-thinking.yml
include_reasoning: true
max_length: 100000000
_kwargs: {fail_on_error: false}'
for n in 10 4; do for k in $(seq 1 $n); do
  printf '# Pairwise window condition (coding, cyber 2026-I): configs/judges/reasoning/kimi-k2-thinking-full.yml showing only characters %d/%d..%d/%d of each transcript. Same items, prompt, orders and reasoning visibility as the full-transcript baseline.\n%s\nwindow: keep-%d/%d\n' $((k-1)) $n $k $n "$body" $k $n > windowed/kimi-k2-thinking-keep-${k}of${n}.yml
done; done
for k in 1 2 3 4; do
  printf '# Pairwise window condition (coding, cyber 2026-I): configs/judges/reasoning/kimi-k2-thinking-full.yml with characters %d/4..%d/4 of each transcript removed ("..." in their place). Same items, prompt, orders and reasoning visibility as the full-transcript baseline.\n%s\nwindow: drop-%d/4\n' $((k-1)) $k "$body" $k > windowed/kimi-k2-thinking-drop-${k}of4.yml
done
for mode in only drop; do for c in reasoning text tools; do
  printf '# Pairwise channel condition (coding, cyber 2026-I): configs/judges/reasoning/kimi-k2-thinking-full.yml with %s the %s channel of each transcript (tools = tool calls with their outputs). Same items, prompt and orders as the full-transcript baseline.\n%s\nchannels: %s-%s\n' "$([ $mode = only ] && echo only || echo everything but)" $c "$body" $mode $c > channels/kimi-k2-thinking-${mode}-${c}.yml
done; done
ls windowed channels | wc -l   # 26 including the two READMEs once written
```

- [ ] **Step 2: Check every config loads and lands in its own directory**

```bash
uv run python -c "
from pathlib import Path
from configs.judge import JudgeConfig
labels = {p: JudgeConfig.from_cli(p).condition_label for p in sorted(Path('configs/judges').glob('windowed/*.yml')) + sorted(Path('configs/judges').glob('channels/*.yml'))}
assert len(set(labels.values())) == 24, labels
print(*labels.values(), sep='\n')"
```
Expected: 24 distinct labels, `window-keep-1of10` ... `channels-drop-tools`.

- [ ] **Step 3: READMEs**

`configs/judges/windowed/README.md`:

```
Pairwise window conditions: the Kimi-K2-Thinking reasoning-shown baseline judging only a fraction of
each transcript by characters (keep-k/n) or all but that fraction (drop-k/n). Deciles keep-only,
quartiles keep and drop. Logs land in logs/judge_window-<keep|drop>-<k>of<n>/<domain>/<semester>/kimi-k2-thinking/<hash>.
Design: docs/superpowers/specs/2026-09-16-transcript-windows-and-channels-design.md.
```

`configs/judges/channels/README.md`: the same three lines for `only-<c>` / `drop-<c>` and `logs/judge_channels-<only|drop>-<channel>/`.

In `src/challenges/truncation/README.md`, after the pairwise truncation paragraph: "Position windows and channel ablations of the same baselines: `configs/judges/windowed/`, `configs/judges/channels/`."

- [ ] **Step 4: Commit**

```bash
git add configs/judges/windowed configs/judges/channels src/challenges/truncation/README.md
git commit -m "configs: kimi-k2-thinking window (deciles, quartiles) and channel conditions for 2026-I"
```

---

### Task 5: dry-run preview of every condition, no API calls

**Files:**
- Create (scratchpad, not committed): `<scratchpad>/preview_conditions.py`
- Output: `<scratchpad>/previews/<label>.txt`

**Interfaces:**
- Consumes: `build_dataset` (Task 3), the configs (Task 4), one real pair of coding 2026-I logs.

- [ ] **Step 1: Point the worktree at the data**

The worktree has no `data/` checkout and no `logs/`. Link both to the main checkout without committing the change:

```bash
rmdir data && ln -s ../../../data data && ln -s ../../../logs logs
git update-index --skip-worktree data   # the gitlink now points at a symlink; keep it out of every commit
git status --short   # must not list data or logs
```

The coding 2026-I transcripts must already be LFS-pulled in the main checkout (the truncation runs used them). If `ls -la data/<bench dir>/*.eval` shows pointer stubs, stop and ask; do not pull.

- [ ] **Step 2: Write the preview script**

```python
"""Render one judged pair under every windowed / channel config; no API calls. Throwaway."""
from pathlib import Path
from configs.domain import DomainConfig
from configs.judge import JudgeConfig
from judge.dataset import build_dataset

OUT = Path(__file__).with_name("previews"); OUT.mkdir(exist_ok=True)
MAJOR, MINOR = Path("data/<major log>.eval"), Path("data/<minor log>.eval")  # fill in from data/meta/coding/models.csv path_<bench> columns, 2026-I train
ITEMS = {"<one 2026-I_train item id>"}
for cfg_path in sorted(Path("configs/judges").glob("windowed/*.yml")) + sorted(Path("configs/judges").glob("channels/*.yml")):
    cfg = JudgeConfig.from_cli(cfg_path)
    ds = build_dataset(MAJOR, MINOR, max_len=cfg.max_length, include_reverse=False, include_reasoning=True,
                       items=ITEMS, truncate=cfg.truncate, window=cfg.window, channels=cfg.channels)
    text = ds[0].input if len(ds) else "(pair skipped: window under MIN_RESPONSE_CHARS)"
    (OUT / f"{cfg.condition_label}.txt").write_text(text)
    print(f"{cfg.condition_label:28} chars={len(text):7}")
```

Run: `PYTHONPATH=src uv run python <scratchpad>/preview_conditions.py`

- [ ] **Step 3: Inspect**

Open `previews/channels-only-tools.txt` (only `[Tool Call: ...]` and `[Tool Output (...)]:` lines), `channels-only-reasoning.txt` (only `<THOUGHT_START>` blocks), `window-keep-1of10.txt` vs `window-keep-10of10.txt` (start and end of the transcript), `window-drop-2of4.txt` (`...` in the second quarter). Report the printed character counts to the user with the paths. The user decides whether the rendering is what they want before any run launches.

---

### Task 6: launch, gated on the user's decision

**Files:** none new. Logs land in `logs/judge_window-*/` and `logs/judge_channels-*/`.

- [ ] **Step 1: Present the condition set and wait**

Post, and do not run anything until the user answers:

- Cells: coding 2026-I and cyber 2026-I, Kimi-K2-Thinking, reasoning shown.
- Channels: 6 conditions per domain. Windows: keep deciles (10), keep quartiles (4), drop quartiles (4).
- Verdicts per condition equal the baseline's (cyber 2026-I: 13,744). 24 conditions per domain, 48 runs; the twelve pairwise truncation runs hit the OpenRouter weekly key limit, this is four times that.
- Proposed order: channels first (12 runs), stop and show scores; then quartiles (16); then deciles (20).

- [ ] **Step 2: Run one batch**

```bash
for f in configs/judges/channels/kimi-k2-thinking-*.yml; do for d in coding cyber; do
  uv run evaluate configs/domain/$d.yml --protocol chars --judge $f --semester 2026-I
done; done
```

Run each command in the background and note its output file. `evaluate` collects the missing verdicts and scores the run; an interrupted run resumes at item level (samples with an error are re-judged: the resume fix in `src/protocols/pairwise/judge.py` is on this branch).

- [ ] **Step 3: Check completeness before scoring is trusted**

For each run dir `logs/judge_channels-<mode>-<c>/<domain>/2026-I/kimi-k2-thinking/<hash>/`, count verdicts against the baseline dir `logs/judge/<domain>/2026-I/kimi-k2-thinking/24d6cc32/`:

```bash
uv run python -c "
from pathlib import Path
from protocols.pairwise.judge import from_logs
import sys
for d in sys.argv[1:]:
    v = from_logs(Path(d)); print(d, len(v), v.groupby('bench').size().to_dict())" logs/judge/<domain>/2026-I/kimi-k2-thinking/24d6cc32 logs/judge_channels-*/<domain>/2026-I/kimi-k2-thinking/*
```

A shortfall is either windows under 20 characters (expected, report the count) or a 403 weekly-limit error in the eval logs (stop, tell the user, resume later with the same command).

- [ ] **Step 4: Report**

Per batch: MSE per condition next to the baseline's, retained verdict counts, and which runs are partial. Stop for the user's decision before the next batch.

---

## Self-review

- Spec coverage: scope (Task 6 cells), windows (Tasks 1, 3, 4), channels (Tasks 1, 2, 4), mechanics (Tasks 1-3), cost and launch order (Task 6), out of scope untouched.
- Naming: the module-level function is `window_text`, the parameter and config field `window`; Task 3's test code names `window_text`. Task 1 introduces `parse_window`, `parse_channels`, `CHANNELS`, `condition_label`; Tasks 2-4 use those names.
- Existing hashes: Task 1 keeps `"7ea04f2b"`; Task 2 keeps the full rendering identical (existing renderer tests).
