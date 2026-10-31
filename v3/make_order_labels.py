"""순서 모방 라벨 — **배우는 대상을 "남은 비용" 에서 "띠 안에서 무엇을 고르나" 로 바꾼다.**

왜 바꾸나 (설계안 §17~§19 실측)
--------------------------------
`ĥ`(남은 kg) 회귀로 학습한 priority 는 ε=0.5 에서 **R = 0.637** (기준선보다 느림)이었다.
같은 ε 에서 오라클은 **104.8** — **격차 165배.** 원인은 모형 용량이 아니라 **라벨**이다:

* 라벨이 **최적 경로 위 상태뿐**이었다 (§16.1②).
* 넓은 띠에서 priority 가 실제로 비교하는 후보는 **대부분 경로 밖**이다.
* 경로 위에서만 배운 회귀는 그 바깥에서 **외삽**하고, 외삽이 틀리면 헤맨다.

→ **탐색이 실제로 마주치는 상황을 그대로 문제집으로 쓴다.** 오라클 포컬 탐색을
라우트당 1회 돌리면서, 매 확장에서

    (그 순간 FOCAL 안 후보들의 특징,  오라클이 고른 것이 누구인가)

를 기록한다. 배우는 것이 **정확히 추론 때 쓰이는 결정**이 된다 (§17.3 의 "비용 회귀는
질문에 안 맞는 지표" 의 자연스러운 귀결).

오라클
------
기준선 A\\* 최적 경로 위 상태를 0, 그 밖을 1 로 두는 priority. 경로를 미리 아는 셈이라
**어떤 priority 함수도 이보다 잘할 수 없다** = 상한. 라벨의 "정답" 은 이 오라클의 선택이다.

ε
--
**0.2 · 0.5 로만 만든다.** 좁은 띠(0.01)에서는 오라클조차 1.36 이라 **배울 것이 없다** —
`f_min` 을 올리는 일은 순서와 무관하기 때문이다 (설계안 §18.2).

비용
----
라우트당 **오라클 탐색 1회**뿐이다. 측정 장치(`measure_stage1.py`)가 이미 하는 일이라
새로 비싼 계산이 들어가지 않는다.

    PIPE_SUPPORT_STANDARD=... python3 v3/make_order_labels.py \\
        --labels labels/stage1_sA.jsonl labels/stage1_sB.jsonl \\
        --eps 0.2 0.5 --out labels/order_eps
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.generator import GeneratorConfig, generate                 # noqa: E402
from pipe_routing.layered import (G0, LayeredGraph, clear_cache,             # noqa: E402
                                  heuristic_factory)
from pipe_routing.space import State                                         # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import features, focal, guards, standard_guard                       # noqa: E402

GEN_KW = dict(n_pipes=10)

#: 한 확장에서 기록할 후보 수 상한. |FOCAL| 중앙이 ε=0.01 에서도 3,643 이고 ε=0.5 면
#: 훨씬 크다 — 전부 저장하면 라우트 하나가 수 GB 다. **정답은 반드시 넣고**, 나머지는
#: 무작위 표본. 순위 학습은 "정답 대 오답 몇 개" 만 있으면 되므로 손실이 적다
#: (대조 학습·listwise 어느 쪽이든). 표본 수는 측정 후 조정 (`미결:`).
N_CAND = 16

#: 라우트당 결정 수 상한. 기록 시점 = **오라클이 경로 위 상태를 꺼내는 순간**이므로
#: 자연스러운 상한이 경로 길이(라벨 실측 중앙 28 · 최대 133)다. 긴 경로를 자르지 않게
#: 넉넉히 둔다.
N_STEPS = 160


def oracle_order_labels(g, pipe, path_states, eps: float, budget: int, rng) -> dict | None:
    """오라클 포컬 탐색 1회 → (후보 특징, 정답 인덱스) 표본.

    `focal_search_dual` 과 같은 이중 큐를 쓰되, **FOCAL 안을 들여다봐야** 하므로
    탐색 루프를 여기서 직접 돈다 (측정용 복제 — 전개 규칙은 같다).
    """
    import heapq
    import math

    EPS = focal.EPS
    on_path = {(int(s.node), int(s.dir), round(float(s.run), 3), int(s.bend))
               for s in path_states}

    def key(s):
        return (int(s.node), int(s.dir), round(float(s.run), 3), int(s.bend))

    h = heuristic_factory(g, pipe)
    fm = features.FeatureMaker(g, pipe)
    start = g.start_state(pipe)
    tie = 0
    e0 = (h(start), tie, 0.0, start)
    open_h, pending, foc = [e0], [e0], []
    g_best = {start: 0.0}
    parent = {start: None}
    closed: dict = {}
    expanded = 0
    thr = -math.inf
    samples: list[dict] = []

    def live(ent):
        _f, _t, gv, s = ent
        return gv <= g_best.get(s, math.inf) + EPS and not focal._dominated(closed, s, gv)

    while open_h or foc:
        while open_h and not live(open_h[0]):
            heapq.heappop(open_h)
        if open_h:
            thr = max(thr, open_h[0][0] * (1.0 + eps) + EPS)
        while pending and pending[0][0] <= thr:
            f, t, gv, s = heapq.heappop(pending)
            if gv <= g_best.get(s, math.inf) + EPS:
                heapq.heappush(foc, (0.0 if key(s) in on_path else 1.0, t, f, gv, s))

        s = None
        gv = 0.0
        while foc:
            _k, _t, f, gcur, cand = heapq.heappop(foc)
            if live((f, _t, gcur, cand)):
                s, gv = cand, gcur
                break

        # ── 기록: **오라클이 경로 위 상태를 고르는 순간**이 곧 하나의 결정이다.
        # 처음에는 확장 수에 대해 지수 간격으로 기록했는데 3 라우트에서 결정이 6~9개
        # 뿐이었다 (전부 확장 2~16 지점). 구조적인 이유다 — 오라클은 경로 위 상태에
        # priority 0 을 주므로 **FOCAL 에 들어오는 즉시 꺼낸다.** 그래서 경로 위 상태가
        # 띠 안에 머무는 시간이 사실상 0 이고, 일정에 맞춰 들여다보면 거의 못 잡는다.
        # → 꺼내는 순간에 기록하면 **경로 길이만큼**(라우트당 약 40~130) 나오고,
        # 오답 후보도 그 순간 실제 FOCAL 동거인이라 더 충실하다.
        if s is not None and key(s) in on_path and len(samples) < N_STEPS:
            bad = [(f, gvv, cand) for _k, _t, f, gvv, cand in foc
                   if _k != 0.0 and live((f, 0, gvv, cand))]
            if bad:                                  # 고를 거리가 있을 때만
                others = [bad[i] for i in rng.choice(len(bad), size=min(N_CAND - 1, len(bad)),
                                                     replace=False)]
                chosen = [(0.0, gv, s)] + others
                order = rng.permutation(len(chosen))      # 정답 위치를 섞는다
                # ★ 후보 수는 매번 다르다 (띠가 좁은 순간에는 N_CAND 보다 적다).
                # 배열 모양을 고정해야 쌓이므로 **0 으로 채우고 유효 개수를 적는다** —
                # 학습 손실은 `n_valid` 밖을 가린다 (listwise 표준). 중복으로 채우면
                # 가짜 정답이 생길 수 있어 쓰지 않는다
                X = np.zeros((N_CAND, features.DIM), dtype=np.float32)
                gold = -1
                for slot, idx in enumerate(order):
                    fv, gvv, cand = chosen[idx]
                    hv = (fv - gvv) if idx != 0 else h(s)
                    X[slot] = fm.one(cand, hv)
                    if idx == 0:
                        gold = slot
                samples.append({"expanded": expanded, "n_focal": len(bad) + 1,
                                "X": X, "gold": gold, "n_valid": len(chosen)})

        if s is None:
            if not open_h:
                break
            continue

        closed.setdefault((s.node, s.dir), []).append((s.run, s.bend, s.ends, s.hz, gv))
        expanded += 1
        if g.is_goal(s, pipe):
            return {"status": "ok", "expanded": expanded, "samples": samples}
        if expanded >= budget:
            return {"status": "budget_exhausted", "expanded": expanded, "samples": samples}

        for nb in g.neighbors(s, pipe):
            ng = gv + g.cost(s, nb, pipe)
            if ng + EPS < g_best.get(nb, math.inf) and not focal._dominated(closed, nb, ng):
                g_best[nb] = ng
                parent[nb] = s
                tie += 1
                ent = (ng + h(nb), tie, ng, nb)
                heapq.heappush(open_h, ent)
                if ent[0] <= thr:
                    heapq.heappush(foc, (0.0 if key(nb) in on_path else 1.0, tie, ent[0], ng, nb))
                else:
                    heapq.heappush(pending, ent)

    return {"status": "unreachable", "expanded": expanded, "samples": samples}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="순서 모방 라벨 생성")
    ap.add_argument("--labels", nargs="+", required=True, help="1단계 라벨 .jsonl (최적 경로 출처)")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.2, 0.5])
    ap.add_argument("--out", default="labels/order_eps", help="출력 접두 (ε 별 .npz)")
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--limit", type=int, default=0, help="라우트 수 상한 (시험용)")
    ap.add_argument("--seed", type=int, default=0)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    std = standard_guard.resolve(a.allow_public_standard, what="순서 모방 라벨")

    by_key: dict[tuple, dict] = {}
    for path in a.labels:
        for line in pathlib.Path(path).open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                by_key[(r["scenario"], r["pipe_id"])] = r
    rows = sorted(by_key.values(), key=lambda r: (r["scenario"], r["pipe_id"]))
    guards.assert_seed_range(sorted({r["seed"] for r in rows}), "train")
    if a.limit:
        rows = rows[:a.limit]
    print(f"ok 라우트 {len(rows)} · 시나리오 {len({r['scenario'] for r in rows})} · ε {a.eps}")

    rng = np.random.default_rng(a.seed)
    out_by_eps = {e: {"X": [], "gold": [], "route": [], "n_focal": [], "expanded": [],
                      "size": [], "gravity": [], "n_valid": []} for e in a.eps}
    cur_seed, sc = None, None
    t0 = time.perf_counter()
    for ri, r in enumerate(rows):
        if r["seed"] != cur_seed:
            clear_cache()
            sc = generate(GeneratorConfig(seed=r["seed"], **GEN_KW), r["scenario"])
            cur_seed = r["seed"]
        pipe = next(p for p in sc.pipes if p.id == r["pipe_id"])
        g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
        states = [State(r["node"][k], r["dir"][k], r["run"][k], r["bend"][k],
                        r["ends"][k], r["hz"][k], r["su"][k]) for k in range(r["n_states"])]
        for e in a.eps:
            res = oracle_order_labels(g, pipe, states, e, a.budget, rng)
            if not res or not res["samples"]:
                continue
            d = out_by_eps[e]
            for smp in res["samples"]:
                d["X"].append(smp["X"])
                d["gold"].append(smp["gold"])
                d["route"].append(ri)
                d["n_focal"].append(smp["n_focal"])
                d["expanded"].append(smp["expanded"])
                d["size"].append(r["size"])
                d["gravity"].append(int(r["gravity"]))
                d["n_valid"].append(smp["n_valid"])
        if (ri + 1) % 10 == 0:
            n = {e: len(out_by_eps[e]["gold"]) for e in a.eps}
            print(f"  {ri + 1}/{len(rows)} · 표본 {n} · {round(time.perf_counter() - t0)}s",
                  flush=True)

    outp = pathlib.Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    for e in a.eps:
        d = out_by_eps[e]
        if not d["gold"]:
            print(f"ε={e}: 표본 0 — 저장하지 않음")
            continue
        f = outp.with_name(outp.name + f"_{e}.npz")
        np.savez_compressed(
            f, X=np.stack(d["X"]), gold=np.asarray(d["gold"], dtype=np.int32),
            route=np.asarray(d["route"], dtype=np.int32),
            n_focal=np.asarray(d["n_focal"], dtype=np.int64),
            n_valid=np.asarray(d["n_valid"], dtype=np.int32),
            expanded=np.asarray(d["expanded"], dtype=np.int64),
            size=np.asarray(d["size"]), gravity=np.asarray(d["gravity"], dtype=np.int8),
            names=np.asarray(features.NAMES), eps=np.asarray([e]),
            n_cand=np.asarray([N_CAND]),
            support_standard=np.asarray([json.dumps(std, ensure_ascii=False)]))
        print(f"ε={e}: {f} — 결정 {len(d['gold'])} × 후보 {N_CAND} × {features.DIM} 특징")
    print(f"\n{round(time.perf_counter() - t0)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
