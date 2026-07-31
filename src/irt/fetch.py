"""
Download the Epoch benchmark eval logs.

Selections are based on a leaderboard CSV that maps each Model version to a Logs S3 URL.

The S3 URLs are not publicly readable; each file is served by Epoch's CDN
(logs.epoch.ai), so the host is swapped before downloading.

The CDN sits behind an AWS WAF. Requests carry an aws-waf-token cookie taken from
a browser session, supplied via the EPOCH_WAF_TOKEN environment variable.

Only logs missing from out_dir are downloaded, so re-runs are idempotent.

out_dir lives under external_data/ to distinguish from logs/data collected by us.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd

from draccus_kwargs import wrap

MODEL_COLUMN = "Model version"
LOGS_COLUMN = "Logs"
TIMEOUT = 120
CDN_HOST = "logs.epoch.ai"
WAF_TOKEN_ENV = "EPOCH_WAF_TOKEN"  # aws-waf-token cookie from a browser session
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)


@dataclass
class FetchConfig:
    csv: str = "external_data/epoch-gpqa/gpqa_diamond.csv"
    out_dir: str = "external_data/epoch-gpqa"
    models: Any = "all"  # list of "Model version" names, or "all"
    limit: Optional[int] = None  # cap the download count, for quick test
    kwargs: Optional[dict[str, Any]] = None


def select_models(frame: pd.DataFrame, models: Any
                  )->tuple[list[tuple[str, str]], list[str], list[str]]:
    """
    Find requested models in CSV.

    Returns (resolved, missing_url, not_found), where resolved is the
    list of (name, url) pairs to download.
    """
    available = dict(
        zip(frame[MODEL_COLUMN], frame[LOGS_COLUMN].astype(str).str.strip())
    )
    requested = list(available) if models in (None, "all") else list(models)

    resolved: list[tuple[str, str]] = []
    missing_url: list[str] = [] # no download link
    not_found: list[str] = []   # can't find model at all

    for name in requested:
        if name not in available:
            not_found.append(name)
        elif not available[name]:
            missing_url.append(name)
        else:
            resolved.append((name, available[name]))
    return resolved, missing_url, not_found


def cdn_url(url: str) -> str:
    """Rewrite an S3 log URL to Epoch's CDN host, which actually serves the file."""
    return urlunparse(urlparse(url)._replace(netloc=CDN_HOST))


def download(name: str, url: str, out_dir: Path, token: Optional[str] = None) -> bool:
    """
    Download one log into out_dir if absent.

    Epoch serves logs from a WAF-protected CDN; token is an aws-waf-token from a
    browser session, sent as a cookie so the request clears the WAF.
    """
    target = out_dir / f"{name.replace('/', '_')}.eval"
    if target.exists():
        return False
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Cookie"] = f"aws-waf-token={token}"
    request = Request(cdn_url(url), headers=headers)
    temporary = target.with_suffix(".eval.tmp")
    try:
        with urlopen(request, timeout=TIMEOUT) as response, open(temporary, "wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


@wrap
def main(cfg: FetchConfig) -> int:
    token = os.environ.get(WAF_TOKEN_ENV)
    if not token:
        print(f"warning: ${WAF_TOKEN_ENV} is not set; downloads will hit the WAF CAPTCHA")

    frame = pd.read_csv(cfg.csv, keep_default_na=False)
    resolved, missing_url, not_found = select_models(frame, cfg.models)
    if cfg.limit is not None:
        resolved = resolved[: cfg.limit]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = skipped = 0
    failed: list[tuple[str, str]] = []
    for name, url in resolved:
        try:
            if download(name, url, out_dir, token=token):
                downloaded += 1
                print(f"downloaded {name}")
            else:
                skipped += 1
        except Exception as error:  # noqa: BLE001
            failed.append((name, str(error)))
            print(f"  failed {name}: {error}")

    print(
        f"\n{downloaded} downloaded, {skipped} already present, "
        f"{len(missing_url)} without a log URL, {len(not_found)} not found, "
        f"{len(failed)} failed"
    )
    for name in missing_url:
        print(f"  no log URL: {name}")
    for name in not_found:
        print(f"  not in CSV: {name}")
    return int(bool(not_found or failed))


if __name__ == "__main__":
    main()
