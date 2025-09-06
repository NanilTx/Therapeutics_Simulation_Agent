import argparse
import json
import math
import shutil
import time
from typing import List, Optional

from .api.app import orch, _summarize_plan, _summarize_pipeline
from .config import RANDOM_SEED, LATENT_DIM, DATA_DIR
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


def _bar(v: int, vmax: int, width: int = 24, fill: str = "█") -> str:
    if vmax <= 0:
        return ""
    n = max(0, min(width, int(round(width * (v / float(vmax))))))
    return fill * n + " " * (width - n)


def _print_table(headers: List[str], rows: List[List[str]], width: int) -> None:
    if not rows:
        return
    w = {h: len(h) for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            w[h] = max(w[h], len(str(v)))
    def fmt_row(row: List[str]) -> str:
        return "  ".join(str(v).ljust(w[h]) for h, v in zip(headers, row))
    print(fmt_row(headers))
    print("-" * min(width, sum(w.values()) + 2 * (len(headers) - 1)))
    for r in rows:
        print(fmt_row(r))


# ---------- Pretty printers ----------

def _print_plan(proposals: List[dict], style: _Style, width: int, table_k: int) -> None:
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
    # Tabular preview
    trows = []
    for p in proposals[: max(1, int(table_k))]:
        trows.append([
            str(p.get("target")),
            _dir_arrow(p.get("direction", -1.0)),
            str(p.get("dose")),
            str(p.get("duration")),
        ])
    print()
    print(f"{style.dim}Proposals (first {len(trows)}):{style.reset}")
    _print_table(["target", "dir", "dose", "duration"], trows, width)
    # Distributions (targets / durations / doses)
    try:
        from collections import Counter

        t_counts = Counter(p.get("target") for p in proposals)
        d_counts = Counter(int(p.get("duration")) for p in proposals)
        dose_counts = Counter(p.get("dose") for p in proposals)
        t_max = max(t_counts.values()) if t_counts else 0
        d_max = max(d_counts.values()) if d_counts else 0
        dose_max = max(dose_counts.values()) if dose_counts else 0
        print()
        print(f"{style.dim}Distribution:{style.reset}")
        for t in sorted(t_counts):
            bar = _bar(t_counts[t], t_max)
            print(f"  target {t:<6} {bar} {t_counts[t]}")
        for dur in sorted(d_counts):
            bar = _bar(d_counts[dur], d_max)
            print(f"  duration {str(dur)+'d':<6} {bar} {d_counts[dur]}")
        for dose in sorted(dose_counts):
            bar = _bar(dose_counts[dose], dose_max)
            print(f"  dose {str(dose):<9} {bar} {dose_counts[dose]}")
    except Exception:
        pass


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


def _print_run(out: dict, style: _Style, width: int, topk: _Optional[int], bio_k: int, show_legend: bool) -> None:
    # Header
    print(f"{style.bold}Pipeline Results{style.reset}")
    # Selected
    pick = out.get("selected", {})
    choice = pick.get("choice", {})
    score = pick.get("score")
    arrow = _dir_arrow(choice.get("direction", -1.0))
    print(f"{style.bold}Selected Candidate{style.reset}")
    _print_table(
        ["target", "dir", "dose", "duration", "score"],
        [[
            str(choice.get("target")),
            arrow,
            str(choice.get("dose")),
            str(choice.get("duration")),
            _fmt_float(score),
        ]],
        width,
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
        m = min(max(1, int(bio_k)), d)
        print(f"Biomarker Effects (first {m} of {d})")
        rows = [[str(i + 1), _fmt_float(deltas[i]), _fmt_float(unc[i])] for i in range(m)]
        _print_table(["index", "delta", "uncertainty"], rows, width)

    # Top-K table using CriticAgent scores
    next_best = None
    if sims:
        # Score breakdown for selected
        try:
            import numpy as np

            sel_idx = None
            # Find selected index by matching dict
            for i, p in enumerate(out.get("proposals", [])):
                if p == choice:
                    sel_idx = i
                    break
            if sel_idx is not None:
                eff = float(abs(np.array(sims[sel_idx].get("biomarker_delta", [])).astype(float)).sum())
                unc = float(np.array(sims[sel_idx].get("uncertainty", [])).astype(float).sum()) + 1e-6
                print(
                    f"Score breakdown: effect_sum={_fmt_float(eff)}  uncertainty_sum={_fmt_float(unc)}  ratio={_fmt_float(eff/unc)}"
                )
        except Exception:
            pass

    if topk and sims:
        critic = CriticAgent()
        import numpy as np

        scores = critic.score(sims)
        order = list(np.argsort(scores)[::-1])
        idx = order[: int(topk)]
        if len(order) > 1:
            next_best = float(scores[order[1]])
        print(f"\n{style.bold}Top {len(idx)} Candidates{style.reset}")
        # Column widths
        cols = ["rank", "target", "dir", "dose", "duration", "score"]
        rows = []
        for r, i in enumerate(idx, start=1):
            p = out["proposals"][i]
            rows.append(
                [
                    ("*" if r == 1 else " ") + str(r),
                    str(p.get("target")),
                    _dir_arrow(p.get("direction", -1.0)),
                    str(p.get("dose")),
                    str(p.get("duration")),
                    _fmt_float(scores[i]),
                ]
            )
        _print_table(cols, rows, width)

    if next_best is not None and isinstance(score, (int, float)):
        try:
            gap = float(score) - float(next_best)
            rel = 100.0 * gap / (float(next_best) + 1e-9)
            print(f"Gap to next best: Δ={_fmt_float(gap)} ({_fmt_float(rel,2)}%)")
        except Exception:
            pass

    # Summary line at end
    print()
    print(_wrap(out.get("summary", _summarize_pipeline(out)), width))
    if show_legend:
        print()
        print("Legend: dir ↑ upregulation, ↓ downregulation. Score = effect_sum / uncertainty_sum (higher is better).")


def cmd_plan(n: int, emit_json: bool, no_color: bool, csv_path: Optional[str], show_k: int, show_legend: bool) -> int:
    orch.data.ensure()
    proposals = orch.hypo.propose(n=n)
    summary = _summarize_plan(proposals)
    width = shutil.get_terminal_size((100, 20)).columns
    style = _Style(_supports_color(no_color))
    _print_plan(proposals, style, width, table_k=show_k)
    print()
    print(_wrap(summary, width))
    if show_legend:
        print()
        print("Legend: dir ↑ upregulation, ↓ downregulation.")
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


def cmd_run(n: int, emit_json: bool, no_color: bool, topk: Optional[int], bio_k: int, show_legend: bool) -> int:
    t0 = time.time()
    out = orch.run_pipeline(n=n)
    out["summary"] = _summarize_pipeline(out)
    dt = time.time() - t0
    width = shutil.get_terminal_size((100, 20)).columns
    style = _Style(_supports_color(no_color))
    _print_run(out, style, width, topk, bio_k=bio_k, show_legend=show_legend)
    # Settings footer (table)
    print()
    print("Configuration")
    _print_table(
        ["param", "value"],
        [
            ["n", str(n)],
            ["seed", str(RANDOM_SEED)],
            ["latent_dim", str(LATENT_DIM)],
            ["data_dir", str(DATA_DIR)],
            ["elapsed_s", _fmt_float(dt, 3)],
        ],
        width,
    )
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
    p_plan.add_argument("--show", type=int, default=6, help="Show first K proposals in a table")
    p_plan.add_argument("--no-legend", action="store_true", help="Hide legend/explanations")

    p_run = sub.add_parser("run", help="Run end-to-end pipeline and print a summary")
    p_run.add_argument("-n", type=int, default=6, help="Number of candidates to evaluate")
    p_run.add_argument("--top", type=int, default=5, help="Show top-K candidates by score")
    p_run.add_argument("--json", action="store_true", help="Also print the JSON output")
    p_run.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    p_run.add_argument("--bio", type=int, default=5, help="Show first K biomarker rows for the selected candidate")
    p_run.add_argument("--no-legend", action="store_true", help="Hide legend/explanations")

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
        return cmd_plan(
            n=args.n,
            emit_json=args.json,
            no_color=args.no_color,
            csv_path=args.csv,
            show_k=max(1, int(args.show)),
            show_legend=(not args.no_legend),
        )
    if args.cmd == "run":
        topk = max(1, int(args.top)) if args.top else None
        return cmd_run(
            n=args.n,
            emit_json=args.json,
            no_color=args.no_color,
            topk=topk,
            bio_k=max(1, int(args.bio)),
            show_legend=(not args.no_legend),
        )
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
