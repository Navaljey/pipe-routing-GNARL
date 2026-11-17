"""1단계 비교선 — 컴파일 A* + `rg` × 가중 w 의 시간-품질 곡선 (설계안 §30.7 ①).

변형 (같은 프로세스·같은 그래프·같은 예산 2²²):
  base        skeleton 컴파일 `_route` (w=1, rg 없음) — 기준
  wa<w>       컴파일 + 가중만 (htab ≡ 0)            — `rg` 기여를 떼어 보기 위한 대조
  rg<w>       컴파일 + max(h, rg) × w               — 사전 계산(역방향 Dijkstra) 시간을 벽시계에 더한다
그래프 생성 시간은 양쪽 공통이라 따로 적는다. numba 는 첫 배관 전에 데운다.

배관마다 한 줄씩 .partial.jsonl 에 쓰고 `--resume` 으로 이어 간다 (백그라운드 2시간 제한).

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_frontier_rg.py --seed0 40000000 --n 10 --out out/frontier_rg.json --resume
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

RG_W = (1.0, 1.1, 1.2, 1.5, 2.0)
WA_W = (1.2, 1.5)


def _st(s):
    return "budget_exhausted" if s == "timeout" else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="컴파일 A* + rg × w 시간-품질 곡선 (§30.7 ①)")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std_info = standard_guard.resolve(a.allow_public_standard, what="곡선 측정")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")

    part = pathlib.Path(a.out + ".partial.jsonl")
    rows = []
    if a.resume and part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {(r["scenario"], r["pipe_id"]) for r in rows}
    warmed = False
    for seed in seeds:
        name = f"valid_{seed}"
        clear_cache()
        gc.collect()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            if (name, pipe.id) in done:
                continue
            t = time.perf_counter()
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            row = {"scenario": name, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe), "graph_sec": round(time.perf_counter() - t, 3), "v": {}}
            if not warmed:
                _route(g, pipe, 1e9, 2048)
                route_rg(g, pipe, np.zeros(len(g.pos)), 2048, 1.3)
                warmed = True
            t = time.perf_counter()
            rb = _route(g, pipe, 1e9, a.budget)
            v = {"status": _st(rb.status), "expanded": rb.expanded, "sec": round(time.perf_counter() - t, 4),
                 "pre_sec": 0.0}
            if v["status"] == "ok":
                v["cost"] = rb.J
                J, ok, _ = _verify_wp(sc, pipe, rb, std)
                v.update(J_verifier=J, verifier_success=ok)
            row["v"]["base"] = v
            t = time.perf_counter()
            dist, _ = relaxed_dist(g, pipe)
            pre = time.perf_counter() - t
            zeros = np.zeros(len(g.pos))
            for kind, ws in (("wa", WA_W), ("rg", RG_W)):
                for w in ws:
                    tab = dist if kind == "rg" else zeros
                    t = time.perf_counter()
                    st, chain, ex, gen = route_rg(g, pipe, tab, a.budget, w)
                    v = {"status": _st(st), "expanded": ex, "sec": round(time.perf_counter() - t, 4),
                         "pre_sec": round(pre, 4) if kind == "rg" else 0.0}
                    if v["status"] == "ok":
                        v["cost"] = _finish_chain(g, pipe, chain, ex, gen, 0.0).J
                        J, ok, _ = verifier_J(sc, g, pipe, chain, std)
                        v.update(J_verifier=J, verifier_success=ok)
                    row["v"][f"{kind}{w}"] = v
            rows.append(row)
            with part.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            vv = row["v"]
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: " + " | ".join(
                f"{k} {x['status'][:6]} {x['sec'] + x['pre_sec']:.2f}s" for k, x in vv.items()), flush=True)
            del g
        del sc
    summ = summarize(rows)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "budget": a.budget,
                                               "seeds": [seeds[0], seeds[-1]], "summary": summ, "rows": rows},
                                              ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return 0


def _verify_wp(sc, pipe, rb, std):
    from pipe_routing.pipeline import j_breakdown
    from pipe_routing.verifier.core import PipeRoute, verify
    rep = verify(sc, [PipeRoute(pipe.id, rb.waypoints)], std=std)["pipes"][pipe.id]
    J = j_breakdown(pipe, rb.waypoints, rep.fittings, rep.supports, sc)
    return J["total"], bool(rep.success), None


def summarize(rows) -> dict:
    """짝지음 = 기준이 성공한 배관. 변형이 거기서 실패하면 따로 센다 (D77 ② 손실)."""
    keys = list(rows[0]["v"].keys()) if rows else []
    pair = [r for r in rows if r["v"]["base"]["status"] == "ok"]
    out = {"n_routes": len(rows), "base_ok": len(pair)}
    for k in keys:
        x = [r["v"][k] for r in rows]
        s = {"ok": sum(v["status"] == "ok" for v in x),
             "only_ok_vs_base": sum(r["v"][k]["status"] == "ok" and r["v"]["base"]["status"] != "ok" for r in rows),
             "lost_vs_base": sum(r["v"][k]["status"] != "ok" for r in pair),
             "wall_all": round(sum(v["sec"] + v["pre_sec"] for v in x), 2)}
        p = [r for r in pair if r["v"][k]["status"] == "ok"]
        if p:
            jr = [r["v"][k]["J_verifier"] / r["v"]["base"]["J_verifier"] for r in p]
            W = [r["v"]["base"]["sec"] / (r["v"][k]["sec"] + r["v"][k]["pre_sec"]) for r in p]
            s.update(n_paired=len(p),
                     wall_paired=round(sum(r["v"][k]["sec"] + r["v"][k]["pre_sec"] for r in p), 2),
                     wall_base_paired=round(sum(r["v"]["base"]["sec"] for r in p), 2),
                     wall_ratio_median=round(statistics.median(W), 3),
                     n_slower=sum(w < 1 for w in W),
                     sum_J_ratio=round(sum(r["v"][k]["J_verifier"] for r in p)
                                       / sum(r["v"]["base"]["J_verifier"] for r in p), 5),
                     J_ratio_max=round(max(jr), 5),
                     n_J_over_1pct=sum(j > 1.01 for j in jr), n_J_over_5pct=sum(j > 1.05 for j in jr),
                     verifier_fail=sum(not r["v"][k]["verifier_success"] for r in p))
        out[k] = s
    return out


if __name__ == "__main__":
    raise SystemExit(main())
