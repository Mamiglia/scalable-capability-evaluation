"""Run evaluation jobs in parallel child processes with one live progress bar each.

Inspect's own display assumes it owns the terminal, so each child swaps it for a
display that forwards sample counts over a queue, and the parent renders them.
"""

import contextlib
import multiprocessing as mp
import re
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterator, Sequence

import inspect_ai._display.core.active as active_display
from inspect_ai._display.plain.display import PlainDisplay, PlainTaskDisplay
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

MAX_WORKERS = 8  # cap parallel evals so we don't overwhelm the machine or the API

Job = tuple[Callable[..., int], tuple[Any, ...], str]


class _QueueTaskDisplay(PlainTaskDisplay):
    """Task display that reports progress to the parent process instead of printing."""

    def __init__(self, task_with_result, queue, label: str, **kwargs):
        super().__init__(task_with_result, **kwargs)
        self.queue = queue
        self.label = label

    def sample_complete(self, complete: int, total: int) -> None:
        super().sample_complete(complete, total)
        self.queue.put((self.label, complete, total))

    def complete(self, result) -> None:
        super().complete(result)
        self.queue.put((self.label, self.samples_complete, self.samples_total))


class _QueueDisplay(PlainDisplay):
    """Display whose tasks forward their progress over ``queue``; errors go to the job log."""

    def __init__(self, queue, label: str):
        super().__init__()
        self.queue = queue
        self.label = label

    def update_task_count(self, *args, **kwargs) -> None:
        pass

    @contextlib.contextmanager
    def task(self, profile) -> Iterator[PlainTaskDisplay]:
        with super().task(profile) as task_display:
            task_display._print_status = lambda: None
            task_display._print_status_throttled = lambda: None
            yield _QueueTaskDisplay(
                task_display.task,
                self.queue,
                self.label,
                show_task_names=task_display.show_task_names,
                show_model_names=task_display.show_model_names,
            )


def _worker(worker_fn: Callable[..., int], args: tuple[Any, ...], queue, label: str, log_file: Path) -> int:
    """Run one job with its output going to ``log_file``, so only the parent owns the terminal."""
    active_display._active_display = _QueueDisplay(queue, label)

    stdout, stderr = sys.stdout, sys.stderr
    with open(log_file, "w", buffering=1) as out:  # line-buffered: tail -f works while it runs
        sys.stdout = sys.stderr = out
        try:
            code = worker_fn(*args)
            return code if isinstance(code, int) else 0
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def _drain(queue, progress: Progress, bars: dict[str, Any]) -> None:
    """Apply every queued progress update, without blocking."""
    while True:
        try:
            label, complete, total = queue.get_nowait()
        except Empty:
            return
        if label in bars:
            progress.update(bars[label], completed=complete, total=total)


def _finish_bar(progress: Progress, bar_id) -> None:
    """Fill a bar and hide it, so a finished job never looks stuck mid-way."""
    bar = progress.tasks[bar_id]
    progress.update(bar_id, completed=bar.total or 1, total=bar.total or 1, visible=False)


def _report(errors: Sequence[tuple[str, str, str]], total: int) -> None:
    if not errors:
        print(f"✓ All {total} subtask(s) completed successfully!")
        return
    print("\n" + "=" * 60)
    print(" ❌ SUBTASK FAILURES DETECTED:")
    print("=" * 60)
    for label, reason, output in errors:
        print(f"\n[FAILED SUBTASK]: {label}")
        print(f"  Status: {reason}")
        if output.strip():
            print("\nCaptured Worker Log/Output:")
            print(output.strip())
    print("=" * 60 + "\n")


def eval_runner(jobs: Sequence[Job], max_workers: int = MAX_WORKERS) -> int:
    """Run ``jobs`` (fn, args, label) in parallel; return 0 if all succeeded, else 1."""
    if not jobs:
        return 0

    # spawn, not fork: inspect already has threads running in the parent by now.
    context = mp.get_context("spawn")
    queue = context.Manager().Queue()
    errors: list[tuple[str, str, str]] = []

    log_root = Path(tempfile.mkdtemp(prefix="small_judge_runner_"))
    print(f"Per-job logs in {log_root} (tail -f them to watch a job)")
    log_files = {label: log_root / (re.sub(r"[^\w.-]+", "_", label) + ".log") for _, _, label in jobs}

    with ProcessPoolExecutor(max_workers=max(1, max_workers), mp_context=context) as executor:
        futures = {
            executor.submit(_worker, fn, args, queue, label, log_files[label]): label
            for fn, args, label in jobs
        }

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            bars = {label: progress.add_task(label, total=None) for label in futures.values()}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.1)
                _drain(queue, progress, bars)
                for future in done:
                    _finish_bar(progress, bars[futures[future]])
            _drain(queue, progress, bars)

        for future, label in futures.items():
            exc = future.exception()
            if exc is not None:
                trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                errors.append((label, str(exc), trace))
                continue
            if future.result() != 0:
                errors.append((label, "Subtask evaluation failed", log_files[label].read_text()))

    _report(errors, len(jobs))
    return 1 if errors else 0
