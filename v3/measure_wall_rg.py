"""벽시계 판정 — 컴파일 A\\* (skeleton 기준선) vs 컴파일 A\\* + 완화 그래프 거리 `rg` (설계안 §30.3).

같은 프로세스·같은 그래프·같은 예산 2²² (D73). `rg` 쪽 벽시계 = 사전 계산(역방향 Dijkstra) + 탐색.
그래프 생성 시간은 양쪽 공통이라 빼고 따로 적는다. numba 컴파일 시간은 첫 배관 전에 데워서 뺀다.

판정 (§30.3, 측정 전 선언): 짝지은 라우트에서 배관별 벽시계 비(기준 ÷ rg) 중앙 > 1.0 **그리고** Σ 벽시계 감소,
품질 = Σ검증기 J 비 ≤ 1.01. 기준만 예산 소진 / rg 만 성공한 배관 수를 따로 낸다 (D77 ②).

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_wall_rg.py --seed0 40000000 --n 10 --out out/wall_rg.json
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.astar_rg import route_rg                                         # noqa: E402
from v3.measure_cea import GEN_KW, verifier_J                            # noqa: E402
from v3.measure_controls import relaxed_dist                             # noqa: E402

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,       # noqa: E402
                                  _route, clear_cache)
from pipe_routing.router_astar import _finish_chain                     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="컴파일 A* vs 컴파일 A* + rg 벽시계 (§30.3)")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--repeat", type=int, default=1, help="벽시계 반복 측정 (최솟값 채택)")
    ap.add_argument("--out", required=True)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std_info = standard_guard.resolve(a.allow_public_standard, what="벽시계 측정")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")

    rows, warmed = [], False
    for seed in seeds:
        name = f"valid_{seed}"
        clear_cache()
        gc.collect()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            t = time.perf_counter()
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            row = {"scenario": name, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe), "graph_sec": round(time.perf_counter() - t, 3)}
            if not warmed:                      # numba 컴파일·캐시 적재를 측정 밖으로
                _route(g, pipe, 1e9, 2048)
                route_rg(g, pipe, np.zeros(len(g.pos)), 2048)
                warmed = True
            bs, rs, ps = [], [], []
            for _ in range(a.repeat):
                t = time.perf_counter()
                rb = _route(g, pipe, 1e9, a.budget)
                bs.append(time.perf_counter() - t)
                t = time.perf_counter()
                dist, _ = relaxed_dist(g, pipe)
                ps.append(time.perf_counter() - t)
                t = time.perf_counter()
                st, chain, ex, gen = route_rg(g, pipe, dist, a.budget)
                rs.append(time.perf_counter() - t)
            st_b = "budget_exhausted" if rb.status == "timeout" else rb.status
            st_r = "budget_exhausted" if st == "timeout" else st
            row["base"] = {"status": st_b, "expanded": rb.expanded, "cost": rb.J, "sec": round(min(bs), 4)}
            row["rg"] = {"status": st_r, "expanded": ex, "pre_sec": round(min(ps), 4), "sec": round(min(rs), 4)}
            if st_r == "ok":
                rr = _finish_chain(g, pipe, chain, ex, gen, 0.0)
                row["rg"]["cost"] = rr.J
                J, ok, _ = verifier_J(sc, g, pipe, chain, std)
                row["rg"].update(J_verifier=J, verifier_success=ok)
            if st_b == "ok":
                Jb, okb, _ = _verifier_from_route(sc, g, pipe, rb, std)
                row["base"].update(J_verifier=Jb, verifier_success=okb)
            rows.append(row)
            b, r = row["base"], row["rg"]
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 기준 {b['status'][:6]} {b['expanded']:>9,} {b['sec']:7.3f}s"
                  f" | rg {r['status'][:6]} {r['expanded']:>9,} {r['pre_sec'] + r['sec']:7.3f}s"
                  f" (사전 {r['pre_sec']:.3f})", flush=True)
            del g
        del sc
    summ = summarize(rows)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "budget": a.budget,
                                               "seeds": [seeds[0], seeds[-1]], "summary": summ, "rows": rows},
                                              ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return 0


def _verifier_from_route(sc, g, pipe, rb, std):
    from pipe_routing.pipeline import j_breakdown
    from pipe_routing.verifier.core import PipeRoute, verify
    rep = verify(sc, [PipeRoute(pipe.id, rb.waypoints)], std=std)["pipes"][pipe.id]
    J = j_breakdown(pipe, rb.waypoints, rep.fittings, rep.supports, sc)
    return J["total"], bool(rep.success), rb


def summarize(rows) -> dict:
    pair = [r for r in rows if r["base"]["status"] == "ok" and r["rg"]["status"] == "ok"]
    s = {"n_routes": len(rows), "n_paired": len(pair),
         "base_ok": sum(r["base"]["status"] == "ok" for r in rows),
         "rg_ok": sum(r["rg"]["status"] == "ok" for r in rows),
         "rg_only_ok": sum(r["rg"]["status"] == "ok" and r["base"]["status"] != "ok" for r in rows),
         "base_only_ok": sum(r["base"]["status"] == "ok" and r["rg"]["status"] != "ok" for r in rows),
         "unreachable": sum(r["base"]["status"] == "unreachable" for r in rows)}
    if pair:
        W = [r["base"]["sec"] / (r["rg"]["pre_sec"] + r["rg"]["sec"]) for r in pair]
        R = [r["base"]["expanded"] / max(1, r["rg"]["expanded"]) for r in pair]
        cr = [r["rg"]["cost"] / r["base"]["cost"] for r in pair]
        jp = [r for r in pair if r["base"].get("J_verifier")]
        jr = [r["rg"]["J_verifier"] / r["base"]["J_verifier"] for r in jp]
        wb = sum(r["base"]["sec"] for r in pair)
        wr = sum(r["rg"]["pre_sec"] + r["rg"]["sec"] for r in pair)
        s.update(wall_ratio_median=round(statistics.median(W), 3),
                 wall_ratio_p10=round(float(np.percentile(W, 10)), 3),
                 wall_ratio_p90=round(float(np.percentile(W, 90)), 3),
                 wall_base_sum=round(wb, 2), wall_rg_sum=round(wr, 2),
                 pre_sec_sum=round(sum(r["rg"]["pre_sec"] for r in pair), 2),
                 R_median=round(statistics.median(R), 3), R_total=round(
                     sum(r["base"]["expanded"] for r in pair) / sum(r["rg"]["expanded"] for r in pair), 3),
                 cost_ratio_max=round(max(cr), 6),
                 sum_J_ratio=round(sum(r["rg"]["J_verifier"] for r in jp) / sum(r["base"]["J_verifier"] for r in jp), 5),
                 J_ratio_max=round(max(jr), 5), J_ratio_min=round(min(jr), 5),
                 n_rg_slower=sum(w < 1 for w in W))
        s["gate"] = bool(s["wall_ratio_median"] > 1.0 and wr < wb and s["sum_J_ratio"] <= 1.01)
    return s


if __name__ == "__main__":
    raise SystemExit(main())
