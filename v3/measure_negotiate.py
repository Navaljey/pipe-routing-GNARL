"""3단계 첫 측정 — 라우팅 호출 예산 (설계안 §33.2, 측정 전 선언).

PathFinder 형 협상: 배관마다 빈 맵 그래프(+ 다른 배관 단자 예약) 위에서, 다른 배관의 현재 경로가 표시(D67 항목)를
남기는 엣지에 가격을 더하고 컴파일 A* + rg × 1.1 로 다시 라우팅한다. 다른 배관은 막지 않는다.
충돌이 0 이 되거나 반복 15 에서 멈추고, 남은 충돌 배관만 단단한 장애물로 확정(D50 순서) → 검증기 1회 채점.
기준선 = skeleton `pipeline.run` (순차 첫 해) 같은 맵·같은 프로세스.

변형 (반복 2~): A = 경로 있는 전 배관 · C = 충돌 쌍에 든 배관만. 반복 1(가격 0 = Σ 단독)은 공유.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_negotiate.py --seed0 40100000 --n 1 --out out/negotiate_v1.json --resume
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import pathlib
import resource
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.astar_rg import route_rg                                         # noqa: E402
from v3.measure_controls import relaxed_dist                             # noqa: E402

from pipe_routing import layered                                         # noqa: E402
from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,       # noqa: E402
                                  SupportMembers, clear_cache, terminal_reservations)
from pipe_routing.multi import default_order, routes_conflict           # noqa: E402
from pipe_routing.pipe_spec import pipe_spec                            # noqa: E402
from pipe_routing.router_astar import _finish_chain                     # noqa: E402

W = 1.1                 # §30.8 비교선
MAX_IT = 15
PRES0, PRES_MUL, HIST_ADD = 0.5, 1.5, 0.3   # `추정:` §33.2
GEN_KW = dict(n_pipes=30)                   # 밀집 30 구성 (생성기 기본값 + 30 배관)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _st(s):
    return "budget_exhausted" if s == "timeout" else s


def _key(op, w):
    return ("P", op.id) + tuple(map(tuple, np.asarray(w, dtype=float).tolist()))


class Clock(dict):
    def add(self, k, t):
        self[k] = self.get(k, 0.0) + t


class Negotiator:
    def __init__(self, sc, budget):
        self.sc = sc
        self.budget = budget
        self.pipes = {p.id: p for p in sc.pipes}
        self.order = [p.id for p in default_order(sc)]
        self.res = {pid: terminal_reservations(sc, exclude={pid}) for pid in self.order}
        self.rg = {}
        self.kgpm = {pid: pipe_spec(self.pipes[pid]).kg_per_m for pid in self.order}

    def route(self, pid, paths, factor, clk: Clock, calls: list, it: int):
        """B = pid 를 가격 붙은 빈 그래프에서 라우팅. paths = 지금 다른 배관 경로, factor(A) = pres + hist."""
        B = self.pipes[pid]
        t = time.perf_counter()
        g = LayeredGraph(self.sc, B, (), local_lines=False, support_cost=True, reserved=self.res[pid], grid=G0)
        clk.add("graph", time.perf_counter() - t)
        if pid not in self.rg:
            t = time.perf_counter()
            self.rg[pid], _ = relaxed_dist(g, B)
            clk.add("rg", time.perf_counter() - t)
        t = time.perf_counter()
        n_priced = self._price(g, B, paths, factor)
        clk.add("price", time.perf_counter() - t)
        t = time.perf_counter()
        st, chain, ex, gen = route_rg(g, B, self.rg[pid], self.budget, W)
        sec = time.perf_counter() - t
        clk.add("search", sec)
        st = _st(st)
        wp = _finish_chain(g, B, chain, ex, gen, 0.0).waypoints if st == "ok" else None
        calls.append({"it": it, "pipe": pid, "status": st, "expanded": ex, "search_sec": round(sec, 4),
                      "priced_edges": n_priced})
        del g
        return st, wp

    def _price(self, g, B, paths, factor) -> int:
        F = g.fixed
        tmpl = g._edge_template(F)
        scale = self.kgpm[B.id] / 1000.0          # kg per mm × 배수
        n = 0
        for aid, w in paths.items():
            if aid == B.id or w is None:
                continue
            f = factor(aid)
            if f <= 0:
                continue
            op = self.pipes[aid]
            c = g._item_contrib(F, _key(op, w), op, w)
            for ax in range(3):
                ent = c["axis"][ax]
                if ent is None or not len(ent[0]):
                    continue
                idx = ent[0]
                e = tmpl["axis"][ax]
                a, b, L = e["a"][idx], e["b"][idx], e["L"][idx]
                dp, dm = 2 * ax, 2 * ax + 1
                ok = g.step[a, dp] == b
                p = L[ok] * scale * f
                g.scost[a[ok], dp] += p
                g.scost[b[ok], dm] += p
                n += int(ok.sum())
            for d, ent in c["diag"].items():
                if ent is None or not len(ent[0]):
                    continue
                idx = ent[0]
                e = tmpl["diag"][d]
                a, b, L = e["a"][idx], e["b"][idx], e["L"][idx]
                ok = g.step[a, d] == b
                g.scost[a[ok], d] += L[ok] * scale * f
                n += int(ok.sum())
        return n


def conflicts(neg, paths, clk: Clock):
    t = time.perf_counter()
    ids = [pid for pid in neg.order if paths.get(pid) is not None]
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if routes_conflict(neg.pipes[a], paths[a], neg.pipes[b], paths[b]):
                out.append((a, b))
    clk.add("conflict", time.perf_counter() - t)
    return out


def prune_item_cache(neg, paths):
    live = {_key(neg.pipes[pid], w) for pid, w in paths.items() if w is not None}
    n = 0
    for v in list(layered._FIXED.values()):
        F = v[0] if isinstance(v, tuple) else v
        cache = getattr(F, "__dict__", {}).get("_item_cache")
        if not cache:
            continue
        for ck in [k for k in cache if len(k) > 1 and k[1] == "P" and k[1:] not in live]:
            del cache[ck]
            n += 1
    return n


def negotiate(neg, paths0, failed, variant, clk, calls):
    paths = dict(paths0)
    hist = {}
    pres = PRES0
    trace = []
    conf = conflicts(neg, paths, clk)
    trace.append({"it": 1, "conflict_pairs": len(conf), "rerouted": len(neg.order)})
    it = 1
    while conf and it < MAX_IT:
        it += 1
        bad = {x for pair in conf for x in pair}
        todo = [pid for pid in neg.order if pid not in failed and (variant == "A" or pid in bad)]
        for pid in todo:
            ref = [pid]
            st, wp = neg.route(pid, paths, lambda aid: pres + hist.get(frozenset((aid, ref[0])), 0.0), clk, calls, it)
            if st == "ok":
                paths[pid] = wp
            else:
                failed.add(pid)          # 가격은 엣지를 지우지 않는다 — 상한 소진은 기억하고 다시 부르지 않는다
                paths[pid] = None
        conf = conflicts(neg, paths, clk)
        for a, b in conf:
            k = frozenset((a, b))
            hist[k] = hist.get(k, 0.0) + HIST_ADD
        pres *= PRES_MUL
        trace.append({"it": it, "conflict_pairs": len(conf), "rerouted": len(todo)})
        prune_item_cache(neg, paths)
        print(f"    [{variant}] it {it}: 다시 {len(todo)} · 충돌 쌍 {len(conf)} · 호출 누계 "
              f"{sum(1 for c in calls)} · rss {rss_gb():.1f} GB", flush=True)
    return paths, conf, trace


def commit(neg, paths, conf, clk, calls):
    """남은 충돌 배관을 들어내고 D50 순서로 단단한 장애물 위에서 다시 (rg × 1.1)."""
    t0 = time.perf_counter()
    bad = {x for pair in conf for x in pair}
    final = {pid: w for pid, w in paths.items() if w is not None and pid not in bad}
    members = SupportMembers()
    for pid in neg.order:
        if pid not in bad:
            continue
        P = neg.pipes[pid]
        placed = [(neg.pipes[q], final[q]) for q in neg.order if q in final]
        reserved = terminal_reservations(neg.sc, exclude={pid} | {q for q in final})
        g = LayeredGraph(neg.sc, P, placed, local_lines=False, support_cost=True, reserved=reserved, grid=G0,
                         members=members(neg.sc, placed))
        assert len(g.pos) == len(neg.rg[pid]), "확정 그래프 노드가 다르다"
        t = time.perf_counter()
        st, chain, ex, gen = route_rg(g, P, neg.rg[pid], neg.budget, W)
        st = _st(st)
        calls.append({"it": "commit", "pipe": pid, "status": st, "expanded": ex,
                      "search_sec": round(time.perf_counter() - t, 4)})
        if st == "ok":
            final[pid] = _finish_chain(g, P, chain, ex, gen, 0.0).waypoints
        del g
    clk.add("commit", time.perf_counter() - t0)
    return final, len(bad)


def score(sc, final, std):
    from pipe_routing.pipeline import j_breakdown, required_pipe_ids
    from pipe_routing.verifier.core import PipeRoute, verify
    rep = verify(sc, [PipeRoute(p.id, final.get(p.id) or []) for p in sc.pipes], std=std)["pipes"]
    req, exc = required_pipe_ids(sc)
    ok = {pid for pid, r in rep.items() if r.success}
    J = sum(j_breakdown(sc_p, final[sc_p.id], rep[sc_p.id].fittings, rep[sc_p.id].supports, sc)["total"]
            for sc_p in sc.pipes if sc_p.id in ok)
    return {"full_success": all(pid in ok for pid in req), "required_pipes": len(req),
            "required_passed": sum(pid in ok for pid in req), "passed": len(ok), "routed": len(final),
            "J_total": round(J, 3), "passed_ids": sorted(ok)}


def baseline(sc):
    from pipe_routing.pipeline import run
    t = time.perf_counter()
    out = run(sc, solo_ref=False)
    sec = time.perf_counter() - t
    gl = out["global"]
    return {"full_success": gl["full_success"], "required_pipes": gl["required_pipes"],
            "required_passed": gl["required_passed"], "passed": gl["success_count"], "routed": gl["routed_count"],
            "J_total": gl["J_total"], "wall_sec": round(sec, 2), "computation_time_sec": gl["computation_time_sec"],
            "passed_ids": sorted(r["pipe_id"] for r in out["routes"] if r["success"])}


def run_map(seed, budget, std):
    name = f"valid_{seed}"
    clear_cache()
    gc.collect()
    sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
    row = {"scenario": name, "n_pipes": len(sc.pipes)}

    t = time.perf_counter()
    row["baseline"] = baseline(sc)
    print(f"  {name} 기준선: 필수 {row['baseline']['required_passed']}/{row['baseline']['required_pipes']} · "
          f"J {row['baseline']['J_total']:.0f} · {row['baseline']['wall_sec']:.0f} s", flush=True)
    clear_cache()
    gc.collect()

    neg = Negotiator(sc, budget)
    clk1, calls1 = Clock(), []
    t0 = time.perf_counter()
    paths0, failed0 = {}, set()
    for pid in neg.order:                       # 반복 1 = 가격 0 (Σ 단독)
        st, wp = neg.route(pid, {}, lambda a: 0.0, clk1, calls1, 1)
        paths0[pid] = wp
        if st != "ok":
            failed0.add(pid)
    wall1 = time.perf_counter() - t0
    row["iter1"] = {"wall_sec": round(wall1, 2), "clock": {k: round(v, 2) for k, v in clk1.items()},
                    "failed": {pid: c["status"] for pid, c in zip(neg.order, calls1) if c["status"] != "ok"}}
    print(f"  {name} 반복 1: {wall1:.0f} s · 실패 {len(failed0)}", flush=True)

    for variant in ("C", "A"):
        clk, calls = Clock(clk1), list(calls1)
        t0 = time.perf_counter()
        paths, conf, trace = negotiate(neg, paths0, set(failed0), variant, clk, calls)
        final, n_commit = commit(neg, paths, conf, clk, calls)
        t = time.perf_counter()
        sc_out = score(sc, final, std)
        clk.add("verify", time.perf_counter() - t)
        wall = wall1 + time.perf_counter() - t0
        per_pipe = {}
        for c in calls:
            per_pipe[c["pipe"]] = per_pipe.get(c["pipe"], 0) + 1
        row[variant] = {"wall_sec": round(wall, 2), "clock": {k: round(v, 2) for k, v in clk.items()},
                        "iterations": trace[-1]["it"], "converged": not conf, "trace": trace,
                        "commit_rerouted": n_commit, "calls": len(calls),
                        "calls_per_pipe_median": statistics.median(per_pipe.values()),
                        "calls_per_pipe_max": max(per_pipe.values()),
                        "search_sec_sum": round(sum(c["search_sec"] for c in calls), 2),
                        "score": sc_out, "calls_detail": calls}
        print(f"  {name} [{variant}] 반복 {trace[-1]['it']} · 수렴 {not conf} · 호출 {len(calls)} · "
              f"필수 {sc_out['required_passed']}/{sc_out['required_pipes']} · J {sc_out['J_total']:.0f} · "
              f"{wall:.0f} s", flush=True)
        prune_item_cache(neg, {})
    row["rss_gb_max"] = round(rss_gb(), 2)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3단계 라우팅 호출 예산 (§33)")
    ap.add_argument("--seed0", type=int, default=40_100_000)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std_info = standard_guard.resolve(a.allow_public_standard, what="협상 호출 예산 측정")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")
    part = pathlib.Path(a.out + ".partial.jsonl")
    rows = []
    if a.resume and part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {r["scenario"] for r in rows}
    # numba 데우기 (첫 맵 시간에 섞이지 않게)
    sc0 = generate(GeneratorConfig(seed=seeds[0], n_pipes=2), "warm")
    g0 = LayeredGraph(sc0, sc0.pipes[0], (), local_lines=False, support_cost=True, grid=G0)
    route_rg(g0, sc0.pipes[0], np.zeros(len(g0.pos)), 2048, W)
    del g0, sc0
    for seed in seeds:
        if f"valid_{seed}" in done:
            continue
        row = run_map(seed, a.budget, std)
        row["support_standard"] = std_info.get("sha256")
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        rows.append(row)
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "budget": a.budget, "w": W,
                                               "params": {"max_it": MAX_IT, "pres0": PRES0, "pres_mul": PRES_MUL,
                                                          "hist_add": HIST_ADD},
                                               "rows": rows}, ensure_ascii=False, indent=1, default=float),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
