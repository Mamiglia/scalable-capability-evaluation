"""Utility functions for eval log paths and task_args hashing."""
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict

from inspect_ai._eval.loader import load_tasks
from inspect_ai.log import read_eval_log
from inspect_ai._util.registry import registry_info, registry_lookup

from configs.model import ModelConfig
from configs.task import TaskConfig


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
    b_clean = task.name
    h_str = hash_task_args(task)
    m_clean = model.shortname
    return Path(base_dir) / b_clean / h_str / m_clean

def is_successful_eval(eval_path: Path) -> bool:
    """Check if an evaluation log indicates a successful run."""
    try:
        log = read_eval_log(eval_path, header_only=True)
        return getattr(log, "status", None) == "success"
    except Exception:
        return False


import contextlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from typing import Callable, Iterator, Sequence, Tuple

import inspect_ai._display.core.active as active_module
from inspect_ai._display.plain.display import PlainDisplay, PlainTaskDisplay
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

    @contextlib.contextmanager
    def task(self, profile) -> Iterator[PlainTaskDisplay]:
        with super().task(profile) as task_display:
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


def _eval_runner_worker(worker_fn: Callable[..., int], args: Tuple[Any, ...], queue: Any, label: str) -> int:
    if queue is not None and label is not None:
        attach_queue_display(queue, label)
    return worker_fn(*args)


def eval_runner(jobs: Sequence[Tuple[Callable[..., int], Tuple[Any, ...], str]]) -> int:
    """Execute evaluation jobs in parallel across child processes with live per-subtask progress bars."""
    if not jobs:
        return 0

    manager = mp.Manager()
    queue = manager.Queue()
    labels = [label for _, _, label in jobs]

    with ProcessPoolExecutor(max_workers=max(len(jobs), 1)) as executor:
        futures = [
            executor.submit(_eval_runner_worker, fn, args, queue, label)
            for fn, args, label in jobs
        ]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_ids = {label: progress.add_task(label, total=None) for label in labels}
            
            while not all(f.done() for f in futures):
                while not queue.empty():
                    try:
                        kind, label, complete, total = queue.get_nowait()
                        if label in task_ids:
                            progress.update(task_ids[label], completed=complete, total=total)
                    except Exception:
                        break
                time.sleep(0.05)

            while not queue.empty():
                try:
                    kind, label, complete, total = queue.get_nowait()
                    if label in task_ids:
                        progress.update(task_ids[label], completed=complete, total=total)
                except Exception:
                    break

        results = [f.result() for f in futures]

    return 0 if all(r == 0 for r in results) else 1