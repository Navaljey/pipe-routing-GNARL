"""CEA 휴리스틱 측정 — A\\*(ε=0) 에 `max(h, ĥ)` 를 넣었을 때 확장 수·품질 (설계안 §29).

**무엇을 비교하나.** 같은 그래프·같은 파이썬 A\\* 구현에서 휴리스틱만 바꾼다.

    기준   f = g + h                 (h = skeleton heuristic_factory, admissible — D41)
    CEA    f = g + max(h, h·ê)       (ê = CEA 가 고른 비율 구간 1.05^k)

**왜 포컬(dual)이 아니라 평범한 A\\* 인가 (§29 의 "같은 python dual focal" 을 바꾼 이유).**
`max(h, ĥ)` 는 일관(consistent)하지 않을 수 있다 → `f_min` 이 감소할 수 있다.
`focal_search_dual` 은 문턱 `thr` 를 **비감소 max** 로 들고 있어서, ε=0 에서도 `f_min`
이 내려가면 FOCAL 에 f 가 최소가 아닌 항목이 섞인다 = 그때부터 A\\* 가 아니다.
논문(Futuhi & Sturtevant)이 재는 것은 평범한 A\\* 이므로 여기서도 그것을 쓴다.
재열기(reopen)는 허용한다 — 비일관 h 에서 재열기를 막으면 품질 보장이 더 약해진다.
재열기 횟수를 따로 센다.

**판정 축 (§29 게이트, 사전 선언).** 두 변형이 **모두 푼 라우트**에서
    (a) 라우터 비용 비 ≤ 1.01 **그리고** 검증기 J 비 ≤ 1.01 — **전 라우트**
    (b) 확장 감소 배수(기준/CEA)의 **라우트별 비율 중앙** > 1.36
벽시계는 기록만 한다 — 둘 다 파이썬이라 단가가 기준선 컴파일 A\\* 와 다르다.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_cea.py --ckpt ckpt/cea.npz \\
        --seed0 40000000 --n 3 --out out/cea_measure.json
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import pathlib
import statistics
import sys
import time

import numpy as np

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,       # noqa: E402
                                  _route, heuristic_factory)
from pipe_routing.pipeline import j_breakdown                           # noqa: E402
from pipe_routing.router_astar import _finish_chain                     # noqa: E402
from pipe_routing.verifier.core import PipeRoute, verify                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import features, guards, standard_guard                          # noqa: E402
from v3.focal import EPS, _dominated                                     # noqa: E402
from v3.train_cea import CEAHeuristic                                    # noqa: E402

GEN_KW = dict(n_pipes=10)


def astar_py(space, pipe, h, *, expand_limit: int, time_limit: float):
    """평범한 A\\* — f 순 pop, 지배 가지치기는 `focal.py` 와 같은 규칙, 재열기 허용.

    → (status, cost, chain, expanded, generated, reopened, sec)
    """
    t0 = time.perf_counter()
    start = space.start_state(pipe)
    tie = itertools.count()
    open_h = [(h(start), next(tie), 0.0, start)]
    g_best = {start: 0.0}
    parent = {start: None}
    closed: dict = {}
    closed_states: set = set()
    expanded = generated = reopened = 0
    while open_h:
        f, _t, g, s = heapq.heappop(open_h)
        if g > g_best.get(s, math.inf) + EPS or _dominated(closed, s, g):
            continue
        if s in closed_states:
            reopened += 1
        closed_states.add(s)
        closed.setdefault((s.node, s.dir), []).append((s.run, s.bend, s.ends, s.hz, g))
        expanded += 1
        if space.is_goal(s, pipe):
            chain, cur = [], s
            while cur is not None:
                chain.append(cur)
                cur = parent[cur]
            chain.reverse()
            return "ok", g, chain, expanded, generated, reopened, time.perf_counter() - t0
        if expanded % 2048 == 0 and (expanded >= expand_limit
                                     or time.perf_counter() - t0 > time_limit):
            return "budget_exhausted", None, [], expanded, generated, reopened, time.perf_counter() - t0
        for nb in space.neighbors(s, pipe):
            ng = g + space.cost(s, nb, pipe)
            if ng + EPS < g_best.get(nb, math.inf) and not _dominated(closed, nb, ng):
                g_best[nb] = ng
                parent[nb] = s
                heapq.heappush(open_h, (ng + h(nb), next(tie), ng, nb))
                generated += 1
    return "unreachable", None, [], expanded, generated, reopened, time.perf_counter() - t0


def make_cea_h(cea: CEAHeuristic, g, pipe, hf, stat):
    """max(h, h·ê). 상태별 캐시 — 같은 상태가 g 만 바뀌어 다시 생성되면 다시 추론하지 않는다."""
    fm = features.FeatureMaker(g, pipe)
    cache: dict = {}

    def h(s):
        v = cache.get(s)
        if v is not None:
            return v
        hv = hf(s)
        if hv <= 0.0:
            v = hv
        else:
            t = time.perf_counter()
            r = cea.ratio(fm.one(s, hv))
            stat["sec"] += time.perf_counter() - t
            stat["calls"] += 1
            stat["lifted"] += r > 1.0
            v = hv * r if r > 1.0 else hv
        cache[s] = v
        return v

    return h


def verifier_J(sc, g, pipe, chain, std):
    """경로 → 검증기 채점 (단독). → (J_total, Layer0 성공 여부, 라우터 RouteResult)."""
    rr = _finish_chain(g, pipe, chain, 0, 0, 0.0)
    rep = verify(sc, [PipeRoute(pipe.id, rr.waypoints)], std=std)["pipes"][pipe.id]
    J = j_breakdown(pipe, rr.waypoints, rep.fittings, rep.supports, sc)
    return J["total"], bool(rep.success), rr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CEA 휴리스틱 A* 측정 (검증 시드, 설계안 §29)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--decode", default="argmax", help="argmax | q0.1 | q0.05 …")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--py-budget", type=int, default=600_000, help="파이썬 A* 확장 상한 (양쪽 같음)")
    ap.add_argument("--py-time", type=float, default=180.0, help="변형당 시간 상한 (s)")
    ap.add_argument("--out", required=True)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")          # ★ 평가 시드(20260000~) 로 못 돌린다
    std_info = standard_guard.resolve(a.allow_public_standard, what="CEA 측정")
    cea = CEAHeuristic(a.ckpt, a.decode)
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")

    rows: list[dict] = []
    for seed in seeds:
        name = f"valid_{seed}"
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            hf = heuristic_factory(g, pipe)
            row = {"scenario": name, "seed": seed, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe)}
            rb = _route(g, pipe, 1e9, EXPAND_BUDGET)               # 컴파일 A* — C* 참조
            row["compiled"] = {"status": rb.status, "expanded": rb.expanded, "cost": rb.J}
            if rb.status == "unreachable":
                rows.append(row)
                print(f"  {name}/{pipe.id} {pipe.nominal_size}: 도달불가 — 건너뜀", flush=True)
                continue
            for var in ("base", "cea"):
                stat = {"sec": 0.0, "calls": 0, "lifted": 0}
                h = hf if var == "base" else make_cea_h(cea, g, pipe, hf, stat)
                st, cost, chain, ex, gen, reo, sec = astar_py(
                    g, pipe, h, expand_limit=a.py_budget, time_limit=a.py_time)
                d = {"status": st, "cost": cost, "expanded": ex, "generated": gen,
                     "reopened": reo, "sec": round(sec, 3)}
                if var == "cea":
                    d.update(h_calls=stat["calls"], h_lifted=stat["lifted"],
                             h_sec=round(stat["sec"], 3))
                if st == "ok":
                    J, ok, _ = verifier_J(sc, g, pipe, chain, std)
                    d.update(J_verifier=J, verifier_success=ok)
                row[var] = d
            b, c = row["base"], row["cea"]
            if b["status"] == "ok" and c["status"] == "ok":
                row["R_expanded"] = b["expanded"] / c["expanded"]
                row["cost_ratio"] = c["cost"] / b["cost"]
                # ★ 컴파일 A* 의 C* 대비도 낸다 — 스모크에서 파이썬 A*(기준) 가 C* 보다 0.12%
                # 비싼 라우트가 나왔다 (전개 순서 차이로 지배 가지치기가 다르게 걸린다).
                # 게이트 (a) 는 **둘 중 큰 쪽**으로 본다 (보수적).
                if rb.status == "ok" and rb.J:
                    row["cost_vs_Cstar"] = c["cost"] / rb.J
                    row["base_vs_Cstar"] = b["cost"] / rb.J
                row["J_ratio"] = c["J_verifier"] / b["J_verifier"] if b["J_verifier"] else None
            rows.append(row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 컴파일 {rb.status} {rb.expanded} | "
                  f"기준 {b['status']} {b['expanded']} | CEA {c['status']} {c['expanded']} "
                  f"(재열기 {c['reopened']}) | R {row.get('R_expanded', float('nan')):.3f} "
                  f"비용비 {row.get('cost_ratio', float('nan')):.4f} J비 {row.get('J_ratio') or float('nan'):.4f}",
                  flush=True)

    pair = [r for r in rows if "R_expanded" in r]
    summ = {"n_routes": len(rows), "n_paired": len(pair),
            "base_ok": sum(r.get("base", {}).get("status") == "ok" for r in rows),
            "cea_ok": sum(r.get("cea", {}).get("status") == "ok" for r in rows),
            "cea_only_ok": sum(r.get("cea", {}).get("status") == "ok"
                               and r.get("base", {}).get("status") != "ok" for r in rows),
            "base_only_ok": sum(r.get("base", {}).get("status") == "ok"
                                and r.get("cea", {}).get("status") != "ok" for r in rows)}
    if pair:
        R = [r["R_expanded"] for r in pair]
        cr = [max(r["cost_ratio"], r.get("cost_vs_Cstar", 0.0)) for r in pair]
        bc = [r["base_vs_Cstar"] for r in pair if "base_vs_Cstar" in r]
        jr = [r["J_ratio"] for r in pair if r["J_ratio"] is not None]
        summ.update(
            R_median=round(statistics.median(R), 3),
            R_p10=round(float(np.percentile(R, 10)), 3), R_p90=round(float(np.percentile(R, 90)), 3),
            R_total=round(sum(r["base"]["expanded"] for r in pair)
                          / sum(r["cea"]["expanded"] for r in pair), 3),
            cost_ratio_median=round(statistics.median(cr), 5), cost_ratio_max=round(max(cr), 5),
            base_vs_Cstar_max=round(max(bc), 5) if bc else None,
            J_ratio_median=round(statistics.median(jr), 5) if jr else None,
            J_ratio_max=round(max(jr), 5) if jr else None,
            n_cost_over_1pct=sum(x > 1.01 for x in cr), n_J_over_1pct=sum(x > 1.01 for x in jr),
            reopened_total=sum(r["cea"]["reopened"] for r in pair),
            lifted_frac=round(sum(r["cea"]["h_lifted"] for r in pair)
                              / max(1, sum(r["cea"]["h_calls"] for r in pair)), 3))
        summ["gate_a"] = bool(max(cr) <= 1.01 and (not jr or max(jr) <= 1.01))
        summ["gate_b"] = bool(summ["R_median"] > 1.36)
        summ["gate"] = summ["gate_a"] and summ["gate_b"]
    out = {"support_standard": std_info, "ckpt": a.ckpt, "decode": a.decode,
           "seeds": [seeds[0], seeds[-1]], "py_budget": a.py_budget, "py_time": a.py_time,
           "summary": summ, "rows": rows}
    print("\n" + json.dumps(summ, ensure_ascii=False))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
