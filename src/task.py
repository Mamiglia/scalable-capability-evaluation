from inspect_ai import Task, task
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import answer


from src.dataset import build_dataset, JUDGE_PROMPT_FORMAT

@task
def judge_task(
    log_major_path: str,
    log_minor_path: str,
    format_str: str = JUDGE_PROMPT_FORMAT,
    
):
    log_major = read_eval_log(log_major_path)
    log_minor = read_eval_log(log_minor_path)
    
    name_major = log_major.eval.model
    name_minor = log_minor.eval.model
    benchmark_name = log_major.eval.dataset.name
        
    return Task(
        name=f'{benchmark_name}-{name_major}|{name_minor}',
        description="Judge the responses of two AI models to the same input prompt.",
        dataset=build_dataset(
            log_major_path=log_major_path,
            log_minor_path=log_minor_path,
            format_str=format_str,
        ),
        scorer=answer(),
        metadata={
            "log_major_path": log_major_path,
            "log_minor_path": log_minor_path,
            "format_str": format_str,
            "name_major": name_major,
            "name_minor": name_minor,
            "benchmark_name": benchmark_name,
        }
    )