#!/usr/bin/env python3
"""Fetch GitHub stargazer history and render local star history charts."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "star_data.csv"
ASSETS_DIR = ROOT / "assets"
LIGHT_CHART = ASSETS_DIR / "star_history_light.png"
DARK_CHART = ASSETS_DIR / "star_history_dark.png"


def parse_repo() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY") or os.environ.get("REPO")
    if not repo:
        repo = "zqinli/DiFRa"
    if "/" not in repo:
        raise ValueError(f"Repository must use owner/name format, got: {repo!r}")
    return repo


def parse_link_header(header: str | None) -> dict[str, str]:
    links: dict[str, str] = {}
    if not header:
        return links

    for part in header.split(","):
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url = section[0].strip()[1:-1]
        rel = None
        for item in section[1:]:
            item = item.strip()
            if item.startswith('rel="') and item.endswith('"'):
                rel = item[5:-1]
                break
        if rel:
            links[rel] = url
    return links


def request_json(url: str, token: str | None) -> tuple[list[dict], dict[str, str]]:
    headers = {
        "Accept": "application/vnd.github.star+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "local-star-history-chart",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        payload = response.read().decode("utf-8")
        data = json.loads(payload)

        if remaining == "0" and reset:
            sleep_for = max(0, int(reset) - int(time.time()) + 1)
            print(f"GitHub API rate limit reached; sleeping for {sleep_for}s.")
            time.sleep(sleep_for)

        return data, parse_link_header(response.headers.get("Link"))


def fetch_stargazers(repo: str, token: str | None) -> list[dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/stargazers?per_page=100"
    rows: list[dict[str, str]] = []

    while url:
        try:
            data, links = request_json(url, token)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401 and not token:
                raise RuntimeError(
                    "GitHub API requires authentication for starred_at data. "
                    "Run this in GitHub Actions with GITHUB_TOKEN, or set "
                    "GITHUB_TOKEN locally for manual testing."
                ) from exc
            raise RuntimeError(f"GitHub API request failed: {exc.code} {detail}") from exc

        for item in data:
            user = item.get("user") or {}
            starred_at = item.get("starred_at")
            login = user.get("login")
            html_url = user.get("html_url")
            if starred_at and login:
                rows.append(
                    {
                        "starred_at": starred_at,
                        "user": login,
                        "user_url": html_url or f"https://github.com/{login}",
                    }
                )

        url = links.get("next")

    rows.sort(key=lambda row: (row["starred_at"], row["user"]))
    return rows


def write_if_changed(path: Path, content: bytes) -> bool:
    if path.exists() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def write_csv(rows: list[dict[str, str]]) -> bool:
    lines: list[str] = []
    writer_target = CsvBuffer(lines)
    writer = csv.DictWriter(writer_target, fieldnames=["starred_at", "user", "user_url"])
    writer.writeheader()
    writer.writerows(rows)
    return write_if_changed(DATA_FILE, "".join(lines).encode("utf-8"))


class CsvBuffer:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def write(self, value: str) -> None:
        self.lines.append(value)


def cumulative_by_day(rows: Iterable[dict[str, str]]) -> OrderedDict[datetime, int]:
    totals: OrderedDict[datetime, int] = OrderedDict()
    count = 0
    for row in rows:
        starred_at = datetime.fromisoformat(row["starred_at"].replace("Z", "+00:00"))
        day = datetime(starred_at.year, starred_at.month, starred_at.day, tzinfo=timezone.utc)
        count += 1
        totals[day] = count
    return totals


def style_axes(ax, dark: bool) -> None:
    if dark:
        ax.set_facecolor("#0d1117")
        ax.figure.set_facecolor("#0d1117")
        text = "#f0f6fc"
        grid = "#30363d"
        spine = "#484f58"
    else:
        ax.set_facecolor("#ffffff")
        ax.figure.set_facecolor("#ffffff")
        text = "#24292f"
        grid = "#d0d7de"
        spine = "#d8dee4"

    ax.grid(True, axis="y", color=grid, linewidth=0.9, alpha=0.75)
    ax.grid(True, axis="x", color=grid, linewidth=0.6, alpha=0.35)
    ax.tick_params(colors=text, labelsize=10)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    ax.title.set_color(text)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(spine)


def render_chart(rows: list[dict[str, str]], repo: str, path: Path, dark: bool) -> bool:
    totals = cumulative_by_day(rows)
    color = "#58a6ff" if dark else "#0969da"
    fill = "#1f6feb" if dark else "#54aeff"
    text = "#f0f6fc" if dark else "#24292f"

    fig, ax = plt.subplots(figsize=(10.8, 6.0), dpi=160)
    style_axes(ax, dark)

    if totals:
        dates = list(totals.keys())
        counts = list(totals.values())
        ax.plot(dates, counts, color=color, linewidth=2.8)
        ax.fill_between(dates, counts, color=fill, alpha=0.18)
        ax.scatter(dates[-1], counts[-1], s=42, color=color, zorder=3)
        ax.annotate(
            f"{counts[-1]} stars",
            xy=(dates[-1], counts[-1]),
            xytext=(-8, 12),
            textcoords="offset points",
            ha="right",
            color=text,
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylim(bottom=0, top=max(counts[-1] * 1.12, counts[-1] + 1))
    else:
        placeholder_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        ax.plot([placeholder_date], [0], color=color, linewidth=2.8)
        ax.set_ylim(bottom=0, top=1)
        ax.text(0.5, 0.5, "No star data yet", transform=ax.transAxes, ha="center", color=text)

    ax.set_title(f"{repo} Star History", fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Stars", labelpad=10)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    tmp_path = path.with_suffix(".tmp.png")
    fig.savefig(tmp_path, bbox_inches="tight", facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)

    changed = write_if_changed(path, tmp_path.read_bytes())
    tmp_path.unlink(missing_ok=True)
    return changed


def main() -> int:
    repo = parse_repo()
    token = os.environ.get("GITHUB_TOKEN")
    rows = fetch_stargazers(repo, token)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    changed = [
        write_csv(rows),
        render_chart(rows, repo, LIGHT_CHART, dark=False),
        render_chart(rows, repo, DARK_CHART, dark=True),
    ]

    print(f"Fetched {len(rows)} stars for {repo}.")
    print(f"Updated files: {sum(changed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
