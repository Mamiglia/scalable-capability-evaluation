"""Where in the transcript is the capability information? The thirds and channel analyses (Section 4.2, Appendices
D.8 to D.10).

Writes to <paths.outputs>/transcript_parts/ (config.yaml); each figure as .pdf, .svg and .png with a .txt of every
plotted number and its test, and a .csv of the plotted changes (docs/manuscript_draft.pdf numbering):
    thirds_removed            Figure 4: pairwise judge, each transcript third removed, against the full transcript
    channel_shares            Figure 5A: each channel's share of the characters the models wrote (the .csv: the shares)
    channels_removed          Figure 5B: pairwise judge, each channel removed, against the full transcript
    thirds_kept_alone         Figure 22: pairwise judge, each third kept alone, against the Baseline
    thirds_other_protocols    Figure 23: Deterministic features, Embedding and Absolute scoring, each third removed
    channels_other_protocols  Figure 24: the same protocols, each channel removed
    breakdowns/<name>_errors.csv  every target's squared error per condition, the numbers each figure is drawn from

Reads the pairwise thirds and channel runs named in config.yaml (thirds, channels) from logs/judge/, the published
absolute-scoring challenge runs (logs/direct_truncated-thirds-*, logs/direct_channels-*), <paths.forecasts>/forecasts.csv
(each task's direct judge), the thirds and channel variants in paths.embeddings, the transcripts in paths.data, and the
feature tables scripts/analysis/features.py writes to paths.cache. Launches no judging. Run from the repository root:
    uv run python scripts/analysis/transcript_parts.py [--config scripts/analysis/config.yaml] [--only channels_removed ...]
"""
import argparse
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import loading, stats
from analysis.config import apply_paths, load_config
from analysis.transcript_parts import channels, thirds
from analysis.transcript_parts.paired_change import (baseline_change, domain_changes, figure, grid, method_text,
                                                     number_lines, removal_items)
from capability.task import Task

THIRDS_REMOVED = (
    "Pairwise judge on transcript thirds. Every rendered transcript (reasoning shown) is cut into three equal character "
    "spans and the judge sees the other two with one third replaced by \"...\". Judges, the best reference judge of each "
    "cell: Coding 2025-II MiniMax-M1 and 2026-I Kimi-K2-Thinking, Cyber 2025-II DeepSeek-R1 and 2026-I Kimi-K2-Thinking, "
    "Maths 2025-II DeepSeek-R1 and 2026-I Kimi-K2-Thinking, Science 2025-II MiniMax-M1 and 2026-I MiniMax-M2, each on the "
    "same 40 train items (10 per difficulty quartile). Every condition is scored on the pairs judged in all four "
    "conditions of its cell. Grey point: the Baseline's mean change in squared error from the full transcript, "
    "for scale (untested).")
THIRDS_KEPT_ALONE = (
    "Pairwise judge on single transcript thirds. Every rendered transcript (reasoning shown) is cut into three equal "
    "character spans and the judge sees one third, the others replaced by \"...\". Judges, the best reference judge of "
    "each cell as in the removal figure; the 2026-I thirds are the 40-item runs. Each condition is scored on the pairs "
    "judged in it and the full transcript, on the same 40 train items (10 per difficulty quartile).")
CHANNELS_REMOVED = (
    "Pairwise judge on transcript channels (reasoning, messages, tool calls with their outputs), 40 train items per "
    "semester, 2025-II and 2026-I pooled, each semester's best pairwise judge with reasoning shown. Maths and Science "
    "have no tool calls: the transcript without reasoning is the reasoning-hidden run, and the transcript without "
    "messages is the reasoning alone.")
CHANNEL_SHARES = (
    "Where the models' own words sit in their transcripts, one pie per domain: the share of the characters the models "
    "themselves wrote (role prefixes removed) in their reasoning, their messages to the user, and their tool calls and "
    "tool outputs (Tools), over the correct first-attempt transcripts of the references and targets on the channel "
    "analysis's items (10 per difficulty quartile) of 2025-II and 2026-I. Maths and Science have no tool use.")
THIRDS_OTHER_PROTOCOLS = (
    "Transcript thirds for the Deterministic features, Embedding and Absolute scoring protocols. Every rendered "
    "transcript (reasoning shown) is cut into three equal character spans and one third is removed. Deterministic "
    "features: the 13 features computed on the cut transcript's text and tool calls, standardised ridge refitted on the "
    "references' cut transcripts. Embedding: vectors of the cut transcript, ridge refitted on the references' cut "
    "transcripts. Absolute scoring: the judge of each cell with the lowest reference leave-one-out error rates the cut "
    "transcripts, calibrated on the references' ratings of the same cut, every condition on the ratings present in "
    "every condition (the single thirds' ratings included). 2025-II and 2026-I pooled (16 targets per domain), the same "
    "40 train items (10 per difficulty quartile) as the pairwise removal runs.")
CHANNELS_OTHER_PROTOCOLS = (
    "Transcript channels (reasoning, messages, tool calls with their outputs) for the Deterministic features, Embedding "
    "and Absolute scoring protocols; 2025-II and 2026-I pooled (16 targets per domain), the same 40 train items (10 per "
    "difficulty quartile) as the pairwise removal runs. Maths and Science have no tool calls: without messages is the "
    "reasoning alone. Deterministic features and Embedding are refitted on the references' channel-filtered "
    "transcripts. Embedding: vectors of the transcript rendered without the channel. Deterministic features: the 13 "
    "features computed on the channel-filtered text (the tool-call count is 0 without tools). Absolute scoring: each "
    "task's leave-one-out judge re-rated the channel-removed transcripts, on the same 40 train items, recalibrated on "
    "the same references.")


def removal_tasks(config: dict) -> Iterator[tuple[str, Task, set[str]]]:
    """(domain, task, the task's removal items) for every main-analysis task, in the paper's domain order."""
    items = config["removal_items"]
    for domain in config["tasks"]["domains"]:
        for semester in config["tasks"]["semesters"]:
            (task,) = loading.tasks(loading.domain(domain), [semester])
            print(domain, semester, flush=True)
            yield domain, task, removal_items(task, items["per_quartile"], items["seed"])


def direct_judges(config: dict) -> dict[tuple[str, str], str]:
    """(domain, semester) -> the main analysis's direct judge (lowest reference leave-one-model-out error)."""
    best = stats.best_judges(pd.read_csv(Path(config["paths"]["forecasts"]) / "forecasts.csv"), config["protocols"])
    best = best[best.protocol == "direct"]
    return dict(zip(zip(best.domain, best.semester), best.judge))


def third_rows(config: dict, kept_alone: bool) -> dict[str, str]:
    """Row label -> thirds condition: "Without first third" -> "thirds-23", or kept alone "Only first third" ->
    "thirds-1"."""
    prefix, column = ("Only", 0) if kept_alone else ("Without", 1)
    return {f"{prefix} {label[0].lower()}{label[1:]}": f"thirds-{kept[column]}"
            for label, kept in config["thirds"]["positions"].items()}


def channel_rows(config: dict, tools: bool = True) -> dict[str, str]:
    """Row label -> channel-removed condition; without tools, the domain has no tool calls."""
    return {f"Without {label.lower()}": runs[1] for label, runs in config["channels"]["parts"].items()
            if tools or label != "Tools"}


def channel_colours(config: dict) -> dict[str, str]:
    """Row label ("Without reasoning", ...) -> the channel's colour."""
    return {f"Without {label.lower()}": colour for label, colour in config["display"]["channel_colours"].items()}


def publish(out: Path, name: str, fig: plt.Figure, caption: list[str], table: pd.DataFrame,
            errors: pd.DataFrame) -> None:
    """The figure (.pdf, .svg, .png), its caption (.txt) and plotted changes (.csv), every target's errors (breakdowns/)."""
    (out / "breakdowns").mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(out / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (out / f"{name}.txt").write_text("\n".join(caption) + "\n")
    table.to_csv(out / f"{name}.csv", index=False)
    errors.to_csv(out / "breakdowns" / f"{name}_errors.csv", index=False)
    print("wrote", out / f"{name}.pdf", flush=True)


def pairwise_figure(out: Path, name: str, errors: pd.DataFrame, rows: dict[str, str], against: str, removed: bool,
                 colours: dict[str, str], config: dict, text: str, baseline: bool = False) -> None:
    """A pairwise figure: domain_changes of `errors`, drawn one panel per domain, and its outputs."""
    names = config["display"]["domain_names"]
    table, means = domain_changes(errors, rows, against)
    fig = figure(table, list(rows), removed, colours, names, against, baseline_change(errors) if baseline else None)
    targets = errors[errors.condition == "full"].groupby("domain").size()
    caption = [f"{text} {method_text(against, removed)}", "", *number_lines(table, means, targets, against, names)]
    publish(out, name, fig, caption, table, errors)


def other_protocols(config: dict, variant_prefix: str) -> list[str]:
    """transcript_parts.other_protocols, without Embedding when paths.embeddings has no vectors of the cut transcripts
    (`variant_prefix`: "thirds-" or "channels-"; not yet published on generality-labs/transcript-embeddings)."""
    protocols = list(config["transcript_parts"]["other_protocols"])
    if "embedding" in protocols and not any(Path(config["paths"]["embeddings"]).rglob(f"*.{variant_prefix}*.npz")):
        print(f"no {variant_prefix}* embedding vectors in {config['paths']['embeddings']}: drawn without Embedding", flush=True)
        protocols.remove("embedding")
    return protocols


def other_protocols_figure(out: Path, name: str, errors: pd.DataFrame, rows: dict[str, str], colours: dict[str, str],
                    config: dict, text: str, **layout) -> None:
    """A figure of the other protocols in `errors`, a row of domain panels each, every part removed, and its outputs."""
    names, labels = config["display"]["domain_names"], config["display"]["protocol_names"]
    tables, lines = {}, [f"{text} {method_text('full', True)}"]
    for protocol in [p for p in config["transcript_parts"]["other_protocols"] if p in set(errors.protocol)]:
        part = errors[errors.protocol == protocol]
        tables[protocol], means = domain_changes(part, rows, "full")
        targets = part[part.condition == "full"].groupby("domain").size()
        lines += ["", f"== {labels[protocol]}", "", *number_lines(tables[protocol], means, targets, "full", names)]
    fig = grid({labels[p]: t for p, t in tables.items()}, list(rows), True, colours, names, "full", shared_x=True, **layout)
    table = pd.concat(tables, names=["protocol"]).reset_index(level=0).reset_index(drop=True)
    publish(out, name, fig, lines, table, errors)


def thirds_pairwise_errors(config: dict, kept_alone: bool) -> pd.DataFrame:
    """Every target's squared error under the pairwise thirds runs with each third removed, or kept alone."""
    spec, logs, frames = config["thirds"], Path(config["paths"]["logs"]), []
    for domain, task, keep in removal_tasks(config):
        judge, full, hashes = spec["runs"][domain][task.semester]
        if kept_alone:  # the kept-alone runs of the same judge
            hashes = spec["kept_alone_hashes"][task.semester]
        hashes, run_dir = spec["hashes"][hashes], logs / "judge" / domain / task.semester / judge
        dirs = {"full": run_dir / full} | {c: run_dir / hashes[c.removeprefix("thirds-")]
                                           for c in third_rows(config, kept_alone).values()}
        frames.append(thirds.pairwise_errors(domain, task, dirs, keep, config["protocols"]["pairwise"]))
    return pd.concat(frames, ignore_index=True)


def thirds_removed(config: dict, out: Path) -> None:
    """Figure 4: pairwise judge, each third removed, with the Baseline's change for scale."""
    pairwise_figure(out, "thirds_removed", thirds_pairwise_errors(config, False), third_rows(config, False), "full",
                    True, config["display"]["domain_colours"], config, THIRDS_REMOVED, baseline=True)


def thirds_kept_alone(config: dict, out: Path) -> None:
    """Figure 22: pairwise judge, each third kept alone and the full transcript, against the Baseline."""
    rows = third_rows(config, True) | {"Full transcript": "full"}
    pairwise_figure(out, "thirds_kept_alone", thirds_pairwise_errors(config, True), rows, "easy", False,
                    config["display"]["domain_colours"], config, THIRDS_KEPT_ALONE)


def channels_removed(config: dict, out: Path) -> None:
    """Figure 5B: pairwise judge, each channel removed."""
    spec, logs, frames = config["channels"], Path(config["paths"]["logs"]), []
    for domain, task, keep in removal_tasks(config):
        cell = dict(spec["runs"][domain][task.semester])
        judge, run_dir = cell.pop("judge"), logs / "judge" / domain / task.semester
        train = {c: run_dir / judge / h for c, h in cell.items()}
        if domain in spec["tool_domains"]:
            sources = {c: c for c in channel_rows(config).values()}
        else:  # no tool calls: the reasoning alone is the transcript without messages
            sources = {"drop-text": "only-reasoning"}
        dirs = {c: channels.channel_run_dir(loading.JUDGES / "channels" / f"{judge}-{run}.yml", domain, task.semester)
                for c, run in sources.items()}
        dirs = {c: d for c, d in dirs.items() if d.exists()}
        frames.append(channels.pairwise_errors(domain, task, train, dirs, keep, config["protocols"]["pairwise"]))
    pairwise_figure(out, "channels_removed", pd.concat(frames, ignore_index=True), channel_rows(config), "full", True,
                    channel_colours(config), config, CHANNELS_REMOVED)


def channel_shares(config: dict, out: Path) -> None:
    """Figure 5A: each channel's share of the characters the models wrote, per domain."""
    spec = {label: runs[0] for label, runs in config["channels"]["parts"].items()}  # the transcripts with only that channel
    data, names = Path(config["paths"]["data"]), config["display"]["domain_names"]
    totals = pd.DataFrame(0, index=config["tasks"]["domains"], columns=list(spec))
    for domain, task, keep in removal_tasks(config):
        for label, chars in channels.written_chars(task, keep, spec, data).items():
            totals.loc[domain, label] += chars
    shares = totals.div(totals.sum(axis=1), axis=0)
    out.mkdir(parents=True, exist_ok=True)
    fig = channels.shares_figure(shares, config["display"]["channel_colours"], names)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(out / f"channel_shares{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    shares.to_csv(out / "channel_shares.csv")
    lines = [f"{names[d]}: " + ", ".join(f"{c} {100 * shares.loc[d, c]:.1f}%" for c in shares.columns) + "."
             for d in shares.index]
    (out / "channel_shares.txt").write_text(CHANNEL_SHARES + "\n\n" + "\n".join(lines) + "\n")
    print("wrote", out / "channel_shares.pdf", flush=True)


def thirds_other_protocols(config: dict, out: Path) -> None:
    """Figure 23: Deterministic features, Embedding and Absolute scoring, each third removed."""
    logs, rows = Path(config["paths"]["logs"]), third_rows(config, False)
    removed, alone = list(rows.values()), list(third_rows(config, True).values())
    features = pd.read_csv(Path(config["paths"]["cache"]) / "features_thirds.csv", dtype={"item": str})
    judges, frames, protocols = direct_judges(config), [], other_protocols(config, "thirds-")
    for domain, task, keep in removal_tasks(config):
        for protocol in protocols:
            if protocol == "direct":
                frames.append(thirds.direct_errors(domain, task, judges[domain, task.semester], removed, removed + alone, logs))
            else:
                frames.append(thirds.ridge_errors(protocol, domain, task, keep, ["full", *removed], features))
    other_protocols_figure(out, "thirds_other_protocols", pd.concat(frames, ignore_index=True), rows,
                           config["display"]["domain_colours"], config, THIRDS_OTHER_PROTOCOLS)


def channels_other_protocols(config: dict, out: Path) -> None:
    """Figure 24: Deterministic features, Embedding and Absolute scoring, each channel removed."""
    logs, tool_domains = Path(config["paths"]["logs"]), config["channels"]["tool_domains"]
    reasoning_only = pd.read_csv(Path(config["paths"]["cache"]) / "features_reasoning_only.csv", dtype={"item": str})
    judges, frames, protocols = direct_judges(config), [], other_protocols(config, "channels-")
    for domain, task, keep in removal_tasks(config):
        tools = domain in tool_domains
        removed = list(channel_rows(config, tools).values())
        for protocol in protocols:
            if protocol == "direct":
                frames.append(channels.direct_errors(domain, task, judges[domain, task.semester], removed, keep, logs))
            else:
                frames.append(channels.ridge_errors(protocol, domain, task, keep, ["full", *removed], tools, reasoning_only))
    other_protocols_figure(out, "channels_other_protocols", pd.concat(frames, ignore_index=True), channel_rows(config),
                           channel_colours(config), config, CHANNELS_OTHER_PROTOCOLS, alpha=1, ticks=[-2, -1, 0, 1, 2])  # solid boxes, as Figure 5


ANALYSES = {f.__name__: f for f in (thirds_removed, channel_shares, channels_removed, thirds_kept_alone,
                                     thirds_other_protocols, channels_other_protocols)}


def main(config: dict, only: list[str] | None = None) -> None:
    apply_paths(config)
    out = Path(config["paths"]["outputs"]) / "transcript_parts"
    for name, analysis in ANALYSES.items():
        if only is None or name in only:
            print(f"== {name}", flush=True)
            analysis(config, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="scripts/analysis/config.yaml", help="the analysis configuration")
    parser.add_argument("--only", nargs="+", choices=list(ANALYSES), help="run only these analyses")
    args = parser.parse_args()
    main(load_config(args.config), args.only)
