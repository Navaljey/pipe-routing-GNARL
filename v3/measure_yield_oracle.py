"""양보 판단의 국소 전수 상한 O (설계안 §35.3, 측정 전 선언).

반복 1 (단자 예약을 지킨 단독 경로) 의 충돌 그래프를 연결 묶음으로 나눈다. 크기 ≤ 4 묶음은
"그대로 두는 배관" K = 충돌 그래프의 공집합 아닌 독립 집합 전부를 시도하고, 나머지(양보 배관)는 묶음 안 경로를
모두 걷어낸 뒤 D50 순서로 단단한 장애물(다른 모든 배관의 지금 경로 · 단자 예약 · 부재 예약 D90) 위에서 rg × 1.1 로
다시 깐다. 선택 = (경로 있는 배관 수 ↓, 라우터 비용 합 ↑). 크기 > 4 = 전원 양보 (개수 보고).
조립 후 남은 충돌은 §34 확정과 같은 방식으로 정리 → 검증기 1회 채점.

⚠️ 엄밀한 상한이 아니다 — 양보 배관끼리의 순서는 D50 하나만 본다.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_yield_oracle.py --seed0 40100000 --n 5 --out out/yield_oracle_v1.json --resume
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.astar_rg import route_rg                                         # noqa: E402
from v3.measure_negotiate import (GEN_KW, W, Clock, Negotiator, _st,     # noqa: E402
                                  commit, conflicts, rss_gb, score)

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,       # noqa: E402
                                  SupportMembers, clear_cache, terminal_reservations)
from pipe_routing.router_astar import _finish_chain                     # noqa: E402

MAX_ENUM = 4


def components(ids, pairs):
    par = {i: i for i in ids}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in pairs:
        par[f(a)] = f(b)
    out = {}
    for a, b in pairs:
        out.setdefault(f(a), set()).update((a, b))
    return [sorted(c) for c in out.values()]


def independent_sets(comp, pairs):
    adj = {frozenset(p) for p in pairs}
    for k in range(1, len(comp) + 1):
        for K in itertools.combinations(comp, k):
            if all(frozenset((a, b)) not in adj for a, b in itertools.combinations(K, 2)):
                yield set(K)


def hard_route(neg, pid, paths, members, clk, calls, tag):
    """pid 를 단단한 장애물(paths 의 다른 배관 · 단자 예약 · 부재 예약) 위에서 rg × 1.1 로."""
    P = neg.pipes[pid]
    placed = [(neg.pipes[q], paths[q]) for q in neg.order if q != pid and paths.get(q) is not None]
    reserved = terminal_reservations(neg.sc, exclude={pid} | {q for q, _ in ((p.id, w) for p, w in placed)})
    t = time.perf_counter()
    g = LayeredGraph(neg.sc, P, placed, local_lines=False, support_cost=True, reserved=reserved, grid=G0,
                     members=members(neg.sc, placed))
    clk.add("graph", time.perf_counter() - t)
    assert len(g.pos) == len(neg.rg[pid]), "단단한 그래프 노드가 다르다"
    t = time.perf_counter()
    st, chain, ex, gen = route_rg(g, P, neg.rg[pid], neg.budget, W)
    sec = time.perf_counter() - t
    clk.add("search", sec)
    st = _st(st)
    res = _finish_chain(g, P, chain, ex, gen, 0.0) if st == "ok" else None
    calls.append({"it": tag, "pipe": pid, "status": st, "expanded": ex, "search_sec": round(sec, 4)})
    del g
    return st, (res.waypoints if res else None), (res.J if res else None)


def run_map(seed, budget, std):
    name = f"valid_{seed}"
    sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
    neg = Negotiator(sc, budget)
    clk, calls = Clock(), []
    t0 = time.perf_counter()
    paths, cost, failed = {}, {}, set()
    for pid in neg.order:                       # 반복 1 = §34 와 같은 단독 (가격 0)
        st, wp = neg.route(pid, {}, lambda a: 0.0, clk, calls, 1)
        paths[pid] = wp
        if st != "ok":
            failed.add(pid)
        else:
            cost[pid] = neg.last_J                # 가격 0 → 라우터 비용 그대로
    pairs = conflicts(neg, paths, clk)
    comps = components([p for p in neg.order if paths.get(p) is not None], pairs)
    members = SupportMembers()
    big = [c for c in comps if len(c) > MAX_ENUM]
    log = []
    for comp in comps:
        cpairs = [p for p in pairs if p[0] in comp and p[1] in comp]
        base = {q: w for q, w in paths.items() if q not in comp}
        options = list(independent_sets(comp, cpairs)) if len(comp) <= MAX_ENUM else [set()]
        best = None
        for K in options:
            trial = dict(base)
            for q in K:
                trial[q] = paths[q]
            jsum, n_ok = 0.0, len(K)
            for q in K:
                jsum += cost[q]
            for y in [q for q in neg.order if q in comp and q not in K]:
                st, wp, J = hard_route(neg, y, trial, members, clk, calls, f"opt{len(log)}")
                trial[y] = wp
                if st == "ok":
                    n_ok += 1
                    jsum += J
            key = (-n_ok, jsum)
            log.append({"comp": comp, "keep": sorted(K), "n_ok": n_ok, "router_cost": round(jsum, 3)})
            if best is None or key < best[0]:
                best = (key, {q: trial.get(q) for q in comp}, sorted(K))
        for q, w in best[1].items():
            paths[q] = w
        log.append({"comp": comp, "chosen_keep": best[2], "n_ok": -best[0][0], "router_cost": round(best[0][1], 3)})
        print(f"  {name} 묶음 {comp}: 선택지 {len(options)} → 그대로 {best[2]} · 경로 {-best[0][0]}/{len(comp)}",
              flush=True)
    left = conflicts(neg, paths, clk)
    final, n_commit = commit(neg, paths, left, clk, calls)
    t = time.perf_counter()
    sc_out = score(sc, final, std)
    clk.add("verify", time.perf_counter() - t)
    row = {"scenario": name, "n_pipes": len(sc.pipes), "wall_sec": round(time.perf_counter() - t0, 2),
           "clock": {k: round(v, 2) for k, v in clk.items()}, "iter1_conflict_pairs": len(pairs),
           "components": [len(c) for c in comps], "components_over_enum": len(big),
           "conflicts_after_assembly": len(left), "commit_rerouted": n_commit, "calls": len(calls),
           "options": log, "score": sc_out, "iter1_failed": sorted(failed),
           "final_waypoints": {pid: [[round(c, 3) for c in w] for w in wp] for pid, wp in final.items()},
           "rss_gb_max": round(rss_gb(), 2)}
    print(f"  {name} O: 묶음 {row['components']} (>4: {len(big)}) · 조립 후 충돌 {len(left)} · "
          f"필수 {sc_out['required_passed']}/{sc_out['required_pipes']} · J {sc_out['J_total']:.0f} · "
          f"{row['wall_sec']:.0f} s", flush=True)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="양보 국소 전수 상한 O (§35.3)")
    ap.add_argument("--seed0", type=int, default=40_100_000)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std_info = standard_guard.resolve(a.allow_public_standard, what="양보 상한 측정")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")
    part = pathlib.Path(a.out + ".partial.jsonl")
    rows = []
    if a.resume and part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {r["scenario"] for r in rows}
    for seed in seeds:
        if f"valid_{seed}" in done:
            continue
        clear_cache()
        gc.collect()
        row = run_map(seed, a.budget, std)
        row["support_standard"] = std_info.get("sha256")
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        rows.append(row)
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "budget": a.budget, "w": W,
                                               "max_enum": MAX_ENUM, "rows": rows},
                                              ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
