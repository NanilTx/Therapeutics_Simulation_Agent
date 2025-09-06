import argparse
import json
import math
import shutil
from typing import List, Optional

from .api.app import orch, _summarize_plan, _summarize_pipeline
from .agents.critic_agent import CriticAgent


# ---------- Formatting helpers ----------

def _supports_color(no_color_flag: bool) -> bool:
    if no_color_flag:
        return False
    try:
        import sys

        return sys.stdout.isatty()
    except Exception:
        return False


class _Style:
    def __init__(self, enable: bool) -> None:
        self.enable = enable

    def _code(self, s: str) -> str:
        return s if self.enable else ""

    @property
    def reset(self) -> str:
        return self._code("\033[0m")

    @property
    def bold(self) -> str:
        return self._code("\033[1m")

    @property
    def dim(self) -> str:
        return self._code("\033[2m")

    @property
    def cyan(self) -> str:
        return self._code("\033[36m")

    @property
    def green(self) -> str:
        return self._code("\033[32m")

    @property
    def yellow(self) -> str:
        return self._code("\033[33m")


def _dir_arrow(direction: float) -> str:
    return "↓" if float(direction) < 0 else "↑"


def _fmt_float(x: float, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _wrap(s: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(s, width=width))


# ---------- Pretty printers ----------

def _print_plan(proposals: List[dict], style: _Style, width: int) -> None:
    n = len(proposals)
    if n == 0:
        print("No proposals were generated.")
        return
    targets = sorted({p.get("target") for p in proposals})
    doses = sorted({p.get("dose") for p in proposals})
    durations = sorted({p.get("duration") for p in proposals})
    print(f"{style.bold}Plan Summary{style.reset}")
    print(
        _wrap(
            f"Proposed {n} candidates across targets {targets}, doses {doses}, durations {durations}.",
            width,
        )
    )
    # Show up to 3 examples
    k = min(3, n)
    print(f"{style.dim}Examples ({k}):{style.reset}")
    for p in proposals[:k]:
        arrow = _dir_arrow(p.get("direction", -1.0))
        print(
            f"  - {p.get('target')} {arrow}  dose={p.get('dose')}  duration={p.get('duration')}d"
        )


def _write_csv_proposals(path: str, proposals: List[dict]) -> None:
    import csv, os

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    cols = ["target", "direction", "dose", "duration"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in proposals:
            w.writerow({c: p.get(c) for c in cols})


from typing import Optional as _Optional


def _print_run(out: dict, style: _Style, width: int, topk: _Optional[int]) -> None:
    # Header
    print(f"{style.bold}Pipeline Results{style.reset}")
    # Selected
    pick = out.get("selected", {})
    choice = pick.get("choice", {})
    score = pick.get("score")
    arrow = _dir_arrow(choice.get("direction", -1.0))
    print(
        f"Selected: {style.cyan if hasattr(style,'cyan') else ''}{choice.get('target')}{style.reset} {arrow}  "
        f"dose={choice.get('dose')}  duration={choice.get('duration')}d  "
        f"score={_fmt_float(score)}"
    )

    # Validation
    rmse = out.get("validation", {}).get("rmse")
    print(f"Validation RMSE: {style.green}{_fmt_float(rmse)}{style.reset} (lower is better)")

    # Biomarker effects summary
    sims = out.get("simulations", [])
    if sims:
        deltas = sims[0].get("biomarker_delta", [])
        unc = sims[0].get("uncertainty", [])
        d = len(deltas)
        preview = ", ".join(
            f"{_fmt_float(deltas[i])}±{_fmt_float(unc[i])}" for i in range(min(3, d))
        )
        print(f"Biomarker effects (first {min(3,d)} of {d}): {preview}")

    # Top-K table using CriticAgent scores
    if topk and sims:
        critic = CriticAgent()
        import numpy as np

        scores = critic.score(sims)
        idx = list(np.argsort(scores)[::-1][: int(topk)])
        print(f"\n{style.bold}Top {len(idx)} Candidates{style.reset}")
        # Column widths
        cols = ["rank", "target", "dir", "dose", "duration", "score"]
        rows = []
        for r, i in enumerate(idx, start=1):
            p = out["proposals"][i]
            rows.append(
                [
                    str(r),
                    str(p.get("target")),
                    _dir_arrow(p.get("direction", -1.0)),
                    str(p.get("dose")),
                    str(p.get("duration")),
                    _fmt_float(scores[i]),
                ]
            )
        # Compute widths
        w = {c: len(c) for c in cols}
        for row in rows:
            for c, v in zip(cols, row):
                w[c] = max(w[c], len(v))
        # Print header
        def fmt_row(r: List[str]) -> str:
            return "  ".join(v.ljust(w[c]) for c, v in zip(cols, r))

        print(fmt_row(cols))
        print("-" * min(width, sum(w.values()) + 2 * (len(cols) - 1)))
        for r in rows:
            print(fmt_row(r))

    # Summary line at end
    print()
    print(_wrap(out.get("summary", _summarize_pipeline(out)), width))


def cmd_plan(n: int, emit_json: bool, no_color: bool, csv_path: Optional[str]) -> int:
    orch.data.ensure()
    proposals = orch.hypo.propose(n=n)
    summary = _summarize_plan(proposals)
    width = shutil.get_terminal_size((100, 20)).columns
    style = _Style(_supports_color(no_color))
    _print_plan(proposals, style, width)
    print()
    print(_wrap(summary, width))
    if csv_path:
        try:
            _write_csv_proposals(csv_path, proposals)
            print(f"\nSaved proposals CSV to: {csv_path}")
        except Exception as e:
            print(f"\nFailed to write CSV to {csv_path}: {e}")
    if emit_json:
        print()
        print(json.dumps({"proposals": proposals, "summary": summary}, indent=2))
    return 0


def cmd_run(n: int, emit_json: bool, no_color: bool, topk: Optional[int]) -> int:
    out = orch.run_pipeline(n=n)
    out["summary"] = _summarize_pipeline(out)
    width = shutil.get_terminal_size((100, 20)).columns
    style = _Style(_supports_color(no_color))
    _print_run(out, style, width, topk)
    if emit_json:
        print(json.dumps(out, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Therapeutic Simulation Agent CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Generate candidate proposals and print a summary")
    p_plan.add_argument("-n", type=int, default=6, help="Number of candidates to propose")
    p_plan.add_argument("--json", action="store_true", help="Also print the JSON output")
    p_plan.add_argument("--csv", metavar="PATH", help="Write proposals to a CSV file")
    p_plan.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    p_run = sub.add_parser("run", help="Run end-to-end pipeline and print a summary")
    p_run.add_argument("-n", type=int, default=6, help="Number of candidates to evaluate")
    p_run.add_argument("--top", type=int, default=5, help="Show top-K candidates by score")
    p_run.add_argument("--json", action="store_true", help="Also print the JSON output")
    p_run.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    p_api = sub.add_parser("api", help="Run the FastAPI server (uvicorn)")
    p_api.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_api.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    p_api.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level",
    )
    p_api.add_argument("--workers", type=int, default=1, help="Number of worker processes (reload incompatible)")
    p_api.add_argument("--open-docs", action="store_true", help="Open the Swagger UI in your browser")

    args = parser.parse_args(argv)

    if args.cmd == "plan":
        return cmd_plan(n=args.n, emit_json=args.json, no_color=args.no_color, csv_path=args.csv)
    if args.cmd == "run":
        topk = max(1, int(args.top)) if args.top else None
        return cmd_run(n=args.n, emit_json=args.json, no_color=args.no_color, topk=topk)
    if args.cmd == "api":
        # Import here so other commands don't require uvicorn installed
        try:
            import uvicorn
        except Exception as e:
            print("uvicorn is required for 'api' command. Please install dependencies.")
            return 1
        if (args.workers or 1) > 1 and args.reload:
            print("--workers > 1 cannot be used together with --reload.")
            return 2
        # Optionally open docs in the browser shortly after startup
        if args.open_docs:
            import threading, time, webbrowser

            host_for_url = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
            url = f"http://{host_for_url}:{args.port}/docs"

            def _open():
                try:
                    time.sleep(1.0)
                    webbrowser.open(url)
                except Exception:
                    pass

            threading.Thread(target=_open, daemon=True).start()
        # With reload=True, uvicorn requires an import string, not an app object
        app_path = "tsa.api.app:app"
        uvicorn.run(
            app_path,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=(args.workers if not args.reload else None),
        )
        return 0

    parser.error("Unknown command")
    return 2
