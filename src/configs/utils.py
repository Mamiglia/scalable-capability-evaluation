"""Utility functions for eval log paths and task_args hashing."""
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict
import zipfile

import zstandard

from inspect_ai._eval.loader import load_tasks
from inspect_ai._util.registry import registry_info, registry_lookup

from configs.model import ModelConfig
from configs.task import TaskConfig

MAX_WORKERS = 32  # Limit the number of parallel workers to avoid overwhelming the system


def get_task_args_with_defaults(task: TaskConfig) -> Dict[str, Any]:
    """Load the task from inspect_ai and extract all task arguments including defaults."""
    task_spec = task.task or task.name
    user_args = task.task_args or {}

    fn = registry_lookup("task", task_spec)

    if fn is None:
        try:
            loaded = load_tasks([task_spec], user_args)
            if loaded:
                info = registry_info(loaded[0])
                fn = registry_lookup(info.type, info.name)
        except Exception:
            pass

    if fn is not None:
        try:
            sig = inspect.signature(fn)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_kw:
                bound = sig.bind_partial(**user_args)
            else:
                valid_kwargs = {k: v for k, v in user_args.items() if k in sig.parameters}
                bound = sig.bind_partial(**valid_kwargs)

            bound.apply_defaults()
            res = {}
            for k, v in bound.arguments.items():
                param = sig.parameters.get(k)
                if param and param.kind == inspect.Parameter.VAR_KEYWORD:
                    if isinstance(v, dict):
                        res.update(v)
                else:
                    res[k] = v

            for k, v in user_args.items():
                if k not in res:
                    res[k] = v
            return res
        except Exception:
            pass

    return user_args


def hash_task_args(task: TaskConfig) -> str:
    """Compute a deterministic hash from loaded task arguments."""
    task_args = get_task_args_with_defaults(task)
    s = json.dumps(task_args, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def get_eval_path(
    task: TaskConfig,
    model: ModelConfig,
    base_dir: str | Path = "data",
) -> Path:
    """Return storage path: data/{benchmark}/{task_args_hashing}/{model}"""
    b_clean = task.name.lower()
    h_str = hash_task_args(task)
    m_clean = model.shortname.lower()
    return Path(base_dir) / b_clean / h_str / m_clean

def is_successful_eval(eval_path: Path) -> bool:
    """Check if an evaluation log indicates a successful run."""
    try:
        get_decompressor = zipfile._get_decompressor
        zipfile._get_decompressor = lambda compression: (
            zstandard.ZstdDecompressor().decompressobj()
            if compression == 93
            else get_decompressor(compression)
        )
        with zipfile.ZipFile(eval_path) as archive:
            return json.loads(archive.read("header.json"))["status"] == "success"
    except Exception:
        return False
    finally:
        zipfile._get_decompressor = get_decompressor


import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Callable, Iterator, Sequence, Tuple

import inspect_ai._display.core.active as active_module
from inspect_ai._display.plain.display import PlainDisplay, PlainTaskDisplay
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn


class QueueTaskDisplay(PlainTaskDisplay):
    def __init__(self, task_with_result, queue, task_id, show_task_names=True, show_model_names=True, **kwargs):
        super().__init__(task_with_result, show_task_names=show_task_names, show_model_names=show_model_names, **kwargs)
        self.queue = queue
        self.task_id = task_id

    def sample_complete(self, complete: int, total: int) -> None:
        super().sample_complete(complete, total)
        self.queue.put(("sample", self.task_id, complete, total))

    def complete(self, result) -> None:
        super().complete(result)
        self.queue.put(("complete", self.task_id, self.samples_complete, self.samples_total))


class QueueDisplay(PlainDisplay):
    def __init__(self, queue, task_id):
        super().__init__()
        self.queue = queue
        self.task_id = task_id

    def print(self, *args, **kwargs):
        pass

    def update_task_count(self, *args, **kwargs):
        pass

    def display_error(self, *args, **kwargs):
        pass

    @contextlib.contextmanager
    def task(self, profile) -> Iterator[PlainTaskDisplay]:
        with super().task(profile) as task_display:
            task_display._print_status = lambda: None
            task_display._print_status_throttled = lambda: None
            yield QueueTaskDisplay(
                task_display.task,
                self.queue,
                self.task_id,
                show_task_names=task_display.show_task_names,
                show_model_names=task_display.show_model_names,
            )


def attach_queue_display(queue: Any, task_id: str) -> None:
    if queue is not None:
        active_module._active_display = QueueDisplay(queue, task_id)
        try:
            Console.print = lambda *a, **kw: None
        except Exception:
            pass


def _eval_runner_worker(worker_fn: Callable[..., int], args: Tuple[Any, ...], queue: Any, label: str) -> Tuple[int, str]:
    if queue is not None and label is not None:
        attach_queue_display(queue, label)

    buffer = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buffer
    try:
        res = worker_fn(*args)
        return (res if isinstance(res, int) else 0), buffer.getvalue()
    except Exception:
        err_msg = buffer.getvalue() + "\n" + traceback.format_exc()
        return 1, err_msg
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def eval_runner(jobs: Sequence[Tuple[Callable[..., int], Tuple[Any, ...], str]]) -> int:
    """Execute evaluation jobs in parallel across child processes with live per-subtask progress bars."""
    if not jobs:
        return 0

    manager = mp.Manager()
    queue = manager.Queue()
    labels = [label for _, _, label in jobs]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_eval_runner_worker, fn, args, queue, label): label
            for fn, args, label in jobs
        }
        fut_list = list(futures.keys())

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_ids = {label: progress.add_task(label, total=None) for label in labels}
            finished_tasks = set()

            def _mark_finished(f_done, label_done):
                if label_done in finished_tasks or label_done not in task_ids:
                    return
                finished_tasks.add(label_done)
                tid = task_ids[label_done]
                t_obj = progress.tasks[tid]
                if t_obj.total is None:
                    progress.update(tid, completed=1, total=1, visible=False)
                elif t_obj.completed < t_obj.total:
                    progress.update(tid, completed=t_obj.total, visible=False)
                else:
                    progress.update(tid, visible=False)

            while not all(f.done() for f in fut_list):
                while not queue.empty():
                    try:
                        msg = queue.get_nowait()
                        kind, label, complete, total = msg[:4]
                        if kind in ("sample", "complete") and label in task_ids:
                            progress.update(task_ids[label], completed=complete, total=total)
                    except Exception:
                        break

                for f, label in futures.items():
                    if f.done():
                        _mark_finished(f, label)

                time.sleep(0.05)

            while not queue.empty():
                try:
                    msg = queue.get_nowait()
                    kind, label, complete, total = msg[:4]
                    if kind in ("sample", "complete") and label in task_ids:
                        progress.update(task_ids[label], completed=complete, total=total)
                except Exception:
                    break

            for f, label in futures.items():
                if f.done():
                    _mark_finished(f, label)

        errors = []
        for f, label in futures.items():
            exc = f.exception()
            if exc is not None and isinstance(exc, BaseException):
                tb_str = "".join(traceback.format_exception(type(exc), exc, getattr(exc, "__traceback__", None)))
                errors.append((label, str(exc), tb_str))
            else:
                res = f.result()
                code = res[0] if isinstance(res, tuple) else res
                worker_output = res[1] if isinstance(res, tuple) else ""
                if isinstance(code, int) and code != 0:
                    errors.append((label, f"Subtask evaluation failed with exit code {code}", worker_output))

        if errors:
            print("\n" + "=" * 60)
            print(" ❌ SUBTASK FAILURES DETECTED:")
            print("=" * 60)
            for label, err_msg, output_str in errors:
                print(f"\n[FAILED SUBTASK]: {label}")
                print(f"  Status: {err_msg}")
                if output_str.strip():
                    print("\nCaptured Worker Log/Output:")
                    print(output_str.strip())
            print("=" * 60 + "\n")
        else:
            print(f"✓ All {len(jobs)} subtask(s) completed successfully!")

    return 0 if not errors else 1
