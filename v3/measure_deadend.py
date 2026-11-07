"""Phase 0 — **막다른 길 오라클 상한** (2026-11-08 소유자 승인).

§24 에서 망 단독 보행이 17/17 데드락했다. 원인은 라벨에 막다른 길이 없어
`ĥ` 가 "여기는 끝" 을 말할 출력 자체를 못 가진 것이다 (make_labels.py 는 **최적
경로 위 상태 + 남은 비용**만 저장한다). 그래서 "막다른 길 분류기를 붙이면 되나" 가
남았고, 이 측정이 **분류기를 만들기 전에** 그 천장을 잰다.

**설계 중 정지 조건을 한 번 고쳤다.** 처음엔 *"죽은 구역의 깊이가 얕으면 경고해도
못 피한다 — 300A 는 꺾는 데 18칸이 필요하니까"* 로 잡았는데 **틀렸다.** 죽은 구역의
입구 s 가 DEAD 면 직전 상태 p 는 DEAD 가 아니고, 그것은 **p 에게 DEAD 아닌 다른
후속이 이미 있다**는 뜻이다. 돌려 꺾을 직진이 필요한 게 아니라 p 에서 **다른 갈래**를
고르면 된다. 경고는 한 수 전이어도 유효하다.

그 따름정리: **완벽한 오라클을 끼우면 보행은 막히지 않는다** (ALIVE 상태에는 반드시
ALIVE 후속이 있다). 도달률 100% 는 측정할 게 아니라 정리다.

→ **그래서 재는 것은 J 다.** 완벽한 오라클을 끼운 보행의 J 가 1.01 근처가 아니면
**분류기의 학습 가능성과 무관하게** 이 가지는 죽는다 — 완벽한 분류기보다 잘하는
분류기는 없으므로. §19 의 오라클 상한 측정과 같은 장치다.

    오라클 = "이 상태에서 목표에 닿는 합법 경로가 있는가" 를 그때그때 탐색으로 판정
             (hf 최선우선 + 방문집합. 목표 찾으면 ALIVE / 전선 소진이면 **증명된 DEAD**
              / 상한 도달이면 UNKNOWN → UNKNOWN 은 통과시킨다 = 보수적)
    ⚠️ 오라클은 **불리언만** 쓴다. 찾은 경로를 따라가면 그건 A\\* 이고 측정이 무의미해진다.
    후보는 `ĥ` 순으로 정렬해 **처음 ALIVE 를 만나면 멈춘다** — 그래야 질의가 스텝당
    1~2회로 억제된다 (전수 질의하면 라우트당 수천 번 탐색이다).

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_deadend.py \\
        --ckpt ckpt/h_pilot_h32.pt --seed0 40000000 --n 2 --out out/deadend_phase0.json
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import features, guards, standard_guard                          # noqa: E402
from v3.measure_stage1 import load_net                                   # noqa: E402

GEN_KW = dict(n_pipes=10)


def alive(space, pipe, s, hf, cap: int) -> tuple[str, int]:
    """s 에서 목표에 닿는 합법 경로가 있는가 → ("alive"|"dead"|"unknown", 확장 수).

    `hf` 최선우선 + 방문집합. 전선이 비면 **도달 가능한 전체를 다 봤다는 뜻**이므로
    DEAD 가 증명된다 (비용은 안 보고 도달성만 보므로 지배검사가 필요 없다).
    """
    tie = itertools.count()
    heap = [(hf(s), next(tie), s)]
    seen = {s}
    n = 0
    while heap:
        _h, _t, cur = heapq.heappop(heap)
        n += 1
        if n > cap:
            return "unknown", n
        for nb in space.neighbors(cur, pipe):
            if nb in seen:
                continue
            if space.is_goal(nb, pipe):
                return "alive", n
            seen.add(nb)
            heapq.heappush(heap, (hf(nb), next(tie), nb))
    return "dead", n


def oracle_walk(space, pipe, hf, hhat, *, cap: int, step_limit: int):
    """`ĥ` 로 고르되 오라클이 DEAD 라 증명한 후보는 건너뛰는 보행.

    → dict(status, cost, steps, n_query, n_dead_skipped, n_unknown, oracle_exp)
    """
    s = space.start_state(pipe)
    seen = {s}
    gv = 0.0
    nq = n_dead = n_unk = 0
    oexp = 0
    for step in range(1, step_limit + 1):
        cand = []
        for nb in space.neighbors(s, pipe):
            if nb in seen:
                continue
            ng = gv + space.cost(s, nb, pipe)
            if space.is_goal(nb, pipe):
                return dict(status="ok", cost=ng, steps=step, n_query=nq,
                            n_dead_skipped=n_dead, n_unknown=n_unk, oracle_exp=oexp)
            hv = hf(nb)
            cand.append((hhat(nb, hv), ng, nb))
        if not cand:
            return dict(status="deadlock", cost=None, steps=step, n_query=nq,
                        n_dead_skipped=n_dead, n_unknown=n_unk, oracle_exp=oexp)
        cand.sort(key=lambda t: t[0])
        # ★ 처음 ALIVE 를 만나면 멈춘다. 전수 질의를 하면 라우트당 수천 번 탐색이다
        pick = None
        for _k, ng, nb in cand:
            nq += 1
            verdict, ex = alive(space, pipe, nb, hf, cap)
            oexp += ex
            if verdict == "dead":
                n_dead += 1
                continue
            if verdict == "unknown":
                n_unk += 1            # 증명 못 했으면 통과시킨다 (보수적)
            pick = (ng, nb)
            break
        if pick is None:
            # 모든 후보가 DEAD 로 증명됨 = 지금 상태가 DEAD. 되돌아갈 장치는 없다
            return dict(status="all_dead", cost=None, steps=step, n_query=nq,
                        n_dead_skipped=n_dead, n_unknown=n_unk, oracle_exp=oexp)
        gv, s = pick
        seen.add(s)
    return dict(status="step_limit", cost=None, steps=step_limit, n_query=nq,
                n_dead_skipped=n_dead, n_unknown=n_unk, oracle_exp=oexp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 0 막다른 길 오라클 상한")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--oracle-cap", type=int, default=200_000,
                    help="오라클 1회 질의의 확장 상한. 넘으면 UNKNOWN → 통과")
    ap.add_argument("--step-limit", type=int, default=2_000)
    ap.add_argument("--route-time-limit", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std = standard_guard.resolve(a.allow_public_standard, what="Phase 0 오라클 상한")
    net, ck = load_net(a.ckpt)
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)

    rows: list[dict] = []
    for seed in seeds:
        name = f"valid_{seed}"
        try:
            sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        except Exception as e:
            print(f"  {name}: 생성 실패 {e}", flush=True)
            continue
        for pipe in sc.pipes:
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True,
                             reserved=(), grid=G0)
            hf = heuristic_factory(g, pipe)
            rb = _route(g, pipe, 1e9, a.budget)
            row = {"scenario": name, "seed": seed, "pipe_id": pipe.id,
                   "size": pipe.nominal_size, "gravity": bool(pipe.gravity_pipe),
                   "min_straight": float(getattr(pipe, "min_straight", float("nan"))),
                   "base": {"status": rb.status, "expanded": rb.expanded,
                            "search_sec": round(rb.search_sec, 4), "cost": rb.J}}
            fm = features.FeatureMaker(g, pipe)

            def hhat(s, hv):
                return net((fm.one(s, hv) - mu) / sd, hv)[0]

            t0 = time.perf_counter()
            r = oracle_walk(g, pipe, hf, hhat, cap=a.oracle_cap, step_limit=a.step_limit)
            r["walk_sec"] = round(time.perf_counter() - t0, 2)
            if r["status"] == "ok" and rb.status == "ok" and rb.J:
                r["ratio_cost"] = r["cost"] / rb.J
            row["oracle"] = r
            rows.append(row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 기준 {rb.status} "
                  f"exp {rb.expanded} | 오라클보행 {r['status']} {r['steps']}칸 "
                  f"{r['walk_sec']}s 질의 {r['n_query']} (DEAD 차단 {r['n_dead_skipped']} · "
                  f"UNKNOWN {r['n_unknown']})"
                  + (f" **J비 {r['ratio_cost']:.3f}**" if "ratio_cost" in r else ""), flush=True)

    out = {"support_standard": std, "ckpt": a.ckpt, "seeds": seeds,
           "oracle_cap": a.oracle_cap, "budget": a.budget, "rows": rows}
    solv = [r for r in rows if r["base"]["status"] == "ok"]
    ok = [r for r in solv if r["oracle"]["status"] == "ok"]
    js = [r["oracle"]["ratio_cost"] for r in ok if "ratio_cost" in r["oracle"]]
    out["summary"] = {
        "n_solvable": len(solv), "n_ok": len(ok),
        "reach_rate": round(len(ok) / len(solv), 4) if solv else None,
        # ★ 관문. 완벽한 오라클의 J 가 1.01 근처가 아니면 분류기 가지는 죽는다
        "J_ratio_median": round(statistics.median(js), 5) if js else None,
        "J_ratio_max": round(max(js), 5) if js else None,
        "n_dead_skipped_total": sum(r["oracle"]["n_dead_skipped"] for r in solv),
        "n_unknown_total": sum(r["oracle"]["n_unknown"] for r in solv),
        "n_query_median": round(statistics.median(
            [r["oracle"]["n_query"] for r in solv] or [0]), 1),
        "oracle_exp_median": round(statistics.median(
            [r["oracle"]["oracle_exp"] for r in solv] or [0]), 1),
        "walk_sec_median": round(statistics.median(
            [r["oracle"]["walk_sec"] for r in solv] or [0]), 2),
        "base_sec_median": round(statistics.median(
            [r["base"]["search_sec"] for r in solv] or [0]), 3),
        "status_counts": {s: sum(1 for r in solv if r["oracle"]["status"] == s)
                          for s in sorted({r["oracle"]["status"] for r in solv})},
    }
    print(f"\n요약: {json.dumps(out['summary'], ensure_ascii=False)}", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
