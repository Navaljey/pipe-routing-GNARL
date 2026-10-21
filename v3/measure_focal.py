"""|FOCAL| 측정 — D-q 선행 조건 (설계안 §6-3, 소유자 보고 요구 2026-10-21).

재는 것: 단일 배관(1단계 설정)을 skeleton 표현·비용으로 라우팅하면서, ε 별
`|FOCAL| = #{s ∈ OPEN : f(s) ≤ (1+ε)·f_min}` 의 분포.

왜: |FOCAL| 이 1 에 가까우면 포컬 띠 안에 고를 것이 없고 **신경망 ① 이 할 일이 없다.**
설계 전제가 여기서 갈린다.

    python3 v3/measure_focal.py --set basic --limit 200000 --out out/focal.json

skeleton 은 읽기만 한다. 고치지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pipe_routing.layered import G0, LayeredGraph, heuristic_factory   # noqa: E402
from pipe_routing.scenario import load                                 # noqa: E402

from v3.focal import MeasureSpec, focal_search                         # noqa: E402

SETS = {"basic": ["manual", "procedural"], "manual": ["manual"], "procedural": ["procedural"],
        "stair": ["stair"], "dense": ["dense"]}


def scenario_files(set_name: str) -> list:
    out = []
    for d in SETS[set_name]:
        out += sorted((SK / "scenarios" / d).glob("*.json"))
    return out


def measure_pipe(sc, pipe, spec: MeasureSpec, limit: int):
    """단일 배관 = 다른 배관 없음·예약 없음 (1단계 E1 단일 배관 설정)."""
    t0 = time.perf_counter()
    g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
    build = time.perf_counter() - t0
    r = focal_search(g, pipe, heuristic_factory(g, pipe), eps=0.0, expand_limit=limit, measure=spec)
    rows = r.stats.samples if r.stats else []
    return {
        "pipe_id": pipe.id, "size": pipe.nominal_size, "gravity": bool(pipe.gravity_pipe),
        "status": r.status, "cost": r.cost, "expanded": r.expanded, "generated": r.generated,
        "search_sec": round(r.search_sec, 3), "graph_sec": round(build, 3),
        "measure_sec": round(r.stats.measure_sec, 3) if r.stats else 0.0,
        "n_samples": len(rows),
        "samples": [{"expanded": x["expanded"], "open_viable": x["open_viable"],
                     "dom_checked": x["dom_checked"], "f_min": x["f_min"],
                     "focal": {str(k): v for k, v in x["focal"].items()}} for x in rows],
    }


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[i]


def summarize(routes, eps_list):
    """표본을 모아 ε 별 |FOCAL| 분포. 표본 단위(확장 시점)와 라우트 단위 중앙 둘 다."""
    per_eps = {}
    for e in eps_list:
        k = str(e)
        all_s = [s["focal"][k] for r in routes for s in r["samples"]]
        route_med = [statistics.median([s["focal"][k] for s in r["samples"]])
                     for r in routes if r["samples"]]
        per_eps[k] = {
            "samples": len(all_s),
            "sample_min": min(all_s) if all_s else None,
            "sample_p25": pct(all_s, 25), "sample_median": pct(all_s, 50),
            "sample_p75": pct(all_s, 75), "sample_p90": pct(all_s, 90),
            "sample_max": max(all_s) if all_s else None,
            "sample_mean": round(statistics.fmean(all_s), 2) if all_s else None,
            "frac_eq_1": round(sum(1 for x in all_s if x <= 1) / len(all_s), 4) if all_s else None,
            "frac_ge_2": round(sum(1 for x in all_s if x >= 2) / len(all_s), 4) if all_s else None,
            "frac_ge_5": round(sum(1 for x in all_s if x >= 5) / len(all_s), 4) if all_s else None,
            "route_median_of_median": pct(route_med, 50),
            "route_median_min": min(route_med) if route_med else None,
            "route_median_max": max(route_med) if route_med else None,
        }
    ok = [r for r in routes if r["status"] == "ok"]
    exp = [r["expanded"] for r in routes]
    viable = [s["open_viable"] for r in routes for s in r["samples"]]
    unchecked = sum(1 for r in routes for s in r["samples"] if not s["dom_checked"])
    return {
        "n_routes": len(routes), "n_ok": len(ok),
        "n_budget_exhausted": sum(1 for r in routes if r["status"] == "budget_exhausted"),
        "n_unreachable": sum(1 for r in routes if r["status"] == "unreachable"),
        "expanded_median": pct(exp, 50), "expanded_p90": pct(exp, 90), "expanded_max": max(exp) if exp else None,
        "open_viable_median": pct(viable, 50), "open_viable_max": max(viable) if viable else None,
        "samples_without_dominance_check": unchecked,
        "search_sec_median": pct([r["search_sec"] for r in routes], 50),
        "search_sec_sum": round(sum(r["search_sec"] for r in routes), 1),
        "per_eps": per_eps,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="basic", choices=sorted(SETS))
    ap.add_argument("--limit", type=int, default=200_000, help="측정용 확장 상한 (기준선 2²² 아님)")
    ap.add_argument("--every", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--scenarios", type=int, default=0, help="앞에서 N개만 (0 = 전부)")
    ap.add_argument("--pipes", type=int, default=0, help="시나리오당 앞 N개만 (0 = 전부)")
    ap.add_argument("--out", default="out/focal.json")
    a = ap.parse_args(argv)

    spec = MeasureSpec(every=a.every, max_samples=a.max_samples)
    files = scenario_files(a.set)
    if a.scenarios:
        files = files[:a.scenarios]
    routes, t0 = [], time.perf_counter()
    for f in files:
        sc = load(f)
        pipes = sc.pipes[:a.pipes] if a.pipes else sc.pipes
        for p in pipes:
            row = measure_pipe(sc, p, spec, a.limit)
            row["scenario"] = f.stem
            routes.append(row)
            print(f"  {f.stem:12s} {p.id:5s} {p.nominal_size:>5s} "
                  f"{'T5' if row['gravity'] else 'T1'} {row['status']:17s} "
                  f"exp={row['expanded']:>8,d} open={row['samples'][-1]['open_viable'] if row['samples'] else 0:>7,d} "
                  f"|F|@.01={row['samples'][-1]['focal']['0.01'] if row['samples'] else '-'} "
                  f"({row['search_sec']:.1f}s)", flush=True)
    wall = time.perf_counter() - t0

    out = {"set": a.set, "measure_limit": a.limit, "every": a.every,
           "skeleton_commit": os.popen(f"git -C {SK} rev-parse HEAD").read().strip(),
           "grid": "G0", "setting": "단일 배관 (placed=(), reserve=False), support_cost=True (D57)",
           "note": "|FOCAL| 은 순수 A* 전개 순서에서 쟀다 — 포컬 탐색 자체 순서와는 다를 수 있다",
           "wall_sec": round(wall, 1), "summary": summarize(routes, spec.eps_list), "routes": routes}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    s = out["summary"]
    print(f"\n라우트 {s['n_routes']} (ok {s['n_ok']} · 예산소진 {s['n_budget_exhausted']} · "
          f"도달불가 {s['n_unreachable']})  확장 중앙 {s['expanded_median']:,}  {wall:.0f}s")
    print(f"{'eps':>7} {'중앙':>6} {'p75':>6} {'p90':>6} {'최대':>7} {'=1 비율':>8} {'≥2 비율':>8} {'≥5 비율':>8}")
    for e, d in s["per_eps"].items():
        print(f"{e:>7} {d['sample_median']:>6} {d['sample_p75']:>6} {d['sample_p90']:>6} "
              f"{d['sample_max']:>7} {d['frac_eq_1']:>8.3f} {d['frac_ge_2']:>8.3f} {d['frac_ge_5']:>8.3f}")
    print(f"→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
