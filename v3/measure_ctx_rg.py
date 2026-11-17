"""맥락(놓인 배관) 위에서 `rg` 가 얼마나 느슨해지는가 — 설계안 §30.7 ③.

기준선 순차 배치(D50 순서, rip-up 없음 = 첫 해, ok 면 놓는다)를 skeleton 라우터와 **같은 그래프**로 따라가며
(놓인 배관 · 단자 예약 D59 · 부재 예약 D90), 배관마다:

  base      맥락 그래프 위 컴파일 A* (2²²)  → C*_ctx
  rg_empty  빈 맵 그래프에서 구한 rg 표를 맥락 그래프에 그대로 (협상에서 한 번만 구해 재사용하는 경우)
  rg_ctx    맥락 그래프에서 다시 구한 rg 표 (매 반복 다시 구하는 경우)

느슨함 = C*_ctx / max(h, 표)(start). 놓는 경로 = 기준선 경로 (rg 는 같은 비용이라 배치가 바뀌지 않는다 — 비용 일치를 확인한다).

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_ctx_rg.py --seed0 40000000 --n 10 --out out/ctx_rg.json --resume
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
from v3.measure_cea import GEN_KW                                        # noqa: E402
from v3.measure_controls import relaxed_dist                             # noqa: E402

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,       # noqa: E402
                                  SupportMembers, _route, clear_cache,
                                  heuristic_factory, terminal_reservations)
from pipe_routing.multi import default_order                            # noqa: E402
from pipe_routing.router_astar import _finish_chain                     # noqa: E402


def _st(s):
    return "budget_exhausted" if s == "timeout" else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="맥락 위 rg 느슨함 (§30.7 ③)")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std_info = standard_guard.resolve(a.allow_public_standard, what="맥락 rg 측정")
    from pipe_routing.support_std import load_standard
    if load_standard().info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")

    part = pathlib.Path(a.out + ".partial.jsonl")
    rows = []
    if a.resume and part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done_sc = {r["scenario"] for r in rows if r.get("scenario_done")}
    warmed = False
    for seed in seeds:
        name = f"valid_{seed}"
        if name in done_sc:
            continue
        rows = [r for r in rows if r["scenario"] != name]          # 중간에 끊긴 시나리오는 처음부터
        clear_cache()
        gc.collect()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        members = SupportMembers()
        placed, srows = [], []
        for k, pipe in enumerate(default_order(sc)):
            reserved = terminal_reservations(sc, exclude={pipe.id} | {p.id for p, _ in placed})
            mem = members(sc, placed)
            t = time.perf_counter()
            g = LayeredGraph(sc, pipe, placed, local_lines=False, support_cost=True, reserved=reserved,
                             grid=G0, members=mem)
            gsec = time.perf_counter() - t
            ge = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            assert len(ge.pos) == len(g.pos) and np.array_equal(ge.pos, g.pos), "고정층 노드가 다르다"
            if not warmed:
                _route(g, pipe, 1e9, 2048)
                route_rg(g, pipe, np.zeros(len(g.pos)), 2048)
                warmed = True
            de, pre_e = relaxed_dist(ge, pipe)
            del ge
            t = time.perf_counter()
            rb = _route(g, pipe, 1e9, a.budget)
            bsec = time.perf_counter() - t
            row = {"scenario": name, "pipe_id": pipe.id, "size": pipe.nominal_size, "k_placed": len(placed),
                   "gravity": bool(pipe.gravity_pipe), "graph_sec": round(gsec, 3),
                   "base": {"status": _st(rb.status), "expanded": rb.expanded, "sec": round(bsec, 4),
                            "cost": rb.J if rb.status == "ok" else None}}
            h0 = heuristic_factory(g, pipe)(g.start_state(pipe))
            s0 = g.start_state(pipe).node
            dc, pre_c = relaxed_dist(g, pipe)
            for key, tab, pre in (("rg_empty", de, pre_e), ("rg_ctx", dc, pre_c)):
                t = time.perf_counter()
                st, chain, ex, gen = route_rg(g, pipe, tab, a.budget)
                v = {"status": _st(st), "expanded": ex, "sec": round(time.perf_counter() - t, 4),
                     "pre_sec": round(pre, 4), "h_start": float(max(h0, tab[s0]))}
                if v["status"] == "ok":
                    v["cost"] = _finish_chain(g, pipe, chain, ex, gen, 0.0).J
                row[key] = v
            row["h_start_euclid"] = float(h0)
            if rb.status == "ok":
                placed.append((pipe, rb.waypoints))
            srows.append(row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size} k={len(placed)}: 기준 {row['base']['status'][:6]} "
                  f"{rb.expanded:>9,} {bsec:6.2f}s | empty {row['rg_empty']['expanded']:>9,} "
                  f"{row['rg_empty']['sec']:6.2f}s | ctx {row['rg_ctx']['expanded']:>9,} "
                  f"{row['rg_ctx']['sec'] + row['rg_ctx']['pre_sec']:6.2f}s", flush=True)
            del g
        srows[-1]["scenario_done"] = True
        rows += srows
        with part.open("a", encoding="utf-8") as f:
            for r in srows:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
        del sc, members
    summ = summarize(rows)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "budget": a.budget,
                                               "seeds": [seeds[0], seeds[-1]], "summary": summ, "rows": rows},
                                              ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return 0


def _q(x, p):
    return round(float(np.percentile(x, p)), 3)


def summarize(rows) -> dict:
    ok = [r for r in rows if r["base"]["status"] == "ok"]
    s = {"n_routes": len(rows), "base_ok": len(ok),
         "base_status": {k: sum(r["base"]["status"] == k for r in rows) for k in
                         ("ok", "unreachable", "budget_exhausted")}}
    if not ok:
        return s
    C = np.array([r["base"]["cost"] for r in ok])
    for key in ("rg_empty", "rg_ctx"):
        p = [r for r in ok if r[key]["status"] == "ok"]
        slack = C / np.maximum([r[key]["h_start"] for r in ok], 1e-9)
        W = [r["base"]["sec"] / (r[key]["sec"] + (r[key]["pre_sec"] if key == "rg_ctx" else 0.0)) for r in p]
        R = [r["base"]["expanded"] / max(1, r[key]["expanded"]) for r in p]
        s[key] = {"ok": sum(r[key]["status"] == "ok" for r in rows),
                  "only_ok_vs_base": sum(r[key]["status"] == "ok" and r["base"]["status"] != "ok" for r in rows),
                  "slack_median": round(float(np.median(slack)), 3), "slack_p90": _q(slack, 90),
                  "R_median": round(statistics.median(R), 3) if R else None,
                  "wall_ratio_median": round(statistics.median(W), 3) if W else None,
                  "wall_sum": round(sum(r[key]["sec"] + (r[key]["pre_sec"] if key == "rg_ctx" else 0.0) for r in p), 2),
                  "wall_base_sum": round(sum(r["base"]["sec"] for r in p), 2),
                  "pre_sec_sum": round(sum(r[key]["pre_sec"] for r in p), 2),
                  "cost_ratio_max": round(max(r[key]["cost"] / r["base"]["cost"] for r in p), 6) if p else None}
    sl_e = C / np.maximum([r["h_start_euclid"] for r in ok], 1e-9)
    s["euclid_slack_median"] = round(float(np.median(sl_e)), 3)
    # 놓인 배관 수 구간별 느슨함 (rg_empty 가 맥락이 쌓일수록 얼마나 무뎌지는가)
    by = {}
    for lo, hi in ((0, 1), (1, 4), (4, 7), (7, 99)):
        b = [r for r in ok if lo <= r["k_placed"] < hi]
        if b:
            by[f"k{lo}-{hi - 1}"] = {"n": len(b),
                                     "slack_empty": round(float(np.median([r["base"]["cost"] / r["rg_empty"]["h_start"] for r in b])), 3),
                                     "slack_ctx": round(float(np.median([r["base"]["cost"] / r["rg_ctx"]["h_start"] for r in b])), 3)}
    s["by_k_placed"] = by
    return s


if __name__ == "__main__":
    raise SystemExit(main())
