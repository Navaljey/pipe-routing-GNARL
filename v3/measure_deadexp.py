"""막다른 확장 비율 — "막다른 길 파라미터" 를 **만들기 전** 상한 측정 (2026-11-09 소유자 지시).

질문
----
포컬 탐색은 막다른 길에 빠져도 OPEN 의 다른 후보로 **되돌아 나온다** (§24 의 걷기와 다르다).
그래서 포컬에서 막다른 길의 비용은 실패가 아니라 **헛된 확장 = 시간**이다.

    막다른 확장 비율 d  =  확장한 상태 중, 거기서 목표까지 합법 경로가 **없는** 상태의 비율

막다른 확장을 **전부** 없애는 완벽한 감지기가 있어도 확장 수는 `(1 − d)` 배까지만 준다.

    R 상한 배율  =  1 / (1 − d)

이것이 "막다른 길 파라미터" 아이디어의 **천장**이다. §18 에서 오라클 priority 상한을 먼저
재고 망을 평가한 것, §25 Phase 0 과 같은 방식 — 만들기 전에 천장부터 잰다.

**선언된 관문 (측정 전 고정):** 상한 배율 중앙 < 1.2 (d < 17%) 이면 이 가지를 접는다.

측정 방법
---------
* 검증 시드(40000000~)로 지금 구조 그대로 — 이중 큐 포컬 + 학습 순위 망 (`rank`, ε 0.2) —
  를 돌리고, **확장한 상태**를 저수지 표본으로 K 개 뽑는다 (확장 순서와 무관한 균등 표본).
* 표본마다 `measure_deadend.alive()` (h 순 best-first, 상한 cap) 로 DEAD / ALIVE / UNKNOWN 판정.
  `alive()` 는 `space.neighbors()` 를 쓰므로 **Layer 0 세 예산(직진·서포트 u·중력 hz)을 전부** 본다.
* DEAD 비율 = 하한 (UNKNOWN 은 판정 못 함), DEAD + UNKNOWN = 상한. 둘 다 보고한다.
* 판정은 **상태 단위** — 포컬의 CLOSED·지배와 무관하게 "그 상태에서 갈 길이 있는가" 다.

    PIPE_SUPPORT_STANDARD=/home/claude/pipe-support-standard \\
      python3 v3/measure_deadexp.py --ckpt ckpt/rank_0.2.pt --eps 0.2 --out out/deadexp_0.2.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import time

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (G0, LayeredGraph, _route,             # noqa: E402
                                  heuristic_factory)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import focal, guards, standard_guard                            # noqa: E402
from v3.measure_deadend import alive                                    # noqa: E402
from v3.measure_stage1 import load_net, make_priority                   # noqa: E402

GEN_KW = dict(n_pipes=10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="포컬 탐색 막다른 확장 비율 (상한 측정)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--budget", type=int, default=400_000, help="포컬 확장 상한 (§23 측정과 같게)")
    ap.add_argument("--k", type=int, default=400, help="라우트당 판정할 확장 상태 표본 수")
    ap.add_argument("--cap", type=int, default=3000, help="alive() 확장 상한 (§25.2 와 같은 값)")
    ap.add_argument("--priority-mode", default="rank")
    ap.add_argument("--out", required=True)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")                 # ★ 평가 시드 금지
    std = standard_guard.resolve(a.allow_public_standard, what="막다른 확장 측정")
    net, ck = load_net(a.ckpt)

    rows: list[dict] = []
    for seed in seeds:
        name = f"valid_{seed}"
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            hf = heuristic_factory(g, pipe)
            rb = _route(g, pipe, 1e9, a.budget)
            row = {"scenario": name, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe), "base_status": rb.status,
                   "base_expanded": rb.expanded}
            if rb.status == "unreachable":                  # 시작부터 갈 길이 없다 — 의미 없음
                rows.append(row)
                print(f"  {name}/{pipe.id} {pipe.nominal_size}: 기준 unreachable — 건너뜀", flush=True)
                continue

            # ── 확장 상태 저수지 표본 (Algorithm R, 시드 고정 → 결정적)
            rng = random.Random(seed * 1000 + int(pipe.id[1:]))
            res: list = []
            seen = [0]

            def on_expand(s, _g, _res=res, _seen=seen, _rng=rng, _k=a.k):
                _seen[0] += 1
                if len(_res) < _k:
                    _res.append(s)
                else:
                    j = _rng.randrange(_seen[0])
                    if j < _k:
                        _res[j] = s

            cnt = [0, 0.0]
            pr = make_priority(net, ck, g, pipe, hf, cnt, a.priority_mode)
            t0 = time.perf_counter()
            r = focal.focal_search_dual(g, pipe, hf, eps=a.eps, priority=pr,
                                        expand_limit=a.budget, on_expand=on_expand)
            fsec = time.perf_counter() - t0

            # ── 표본 판정
            t1 = time.perf_counter()
            verdict = {"dead": 0, "alive": 0, "unknown": 0}
            dead_pops: list[int] = []
            for s in res:
                if g.is_goal(s, pipe):
                    verdict["alive"] += 1
                    continue
                v, n = alive(g, pipe, s, hf, a.cap)
                verdict[v] += 1
                if v == "dead":
                    dead_pops.append(n)
            jsec = time.perf_counter() - t1
            k = max(len(res), 1)
            d_lo = verdict["dead"] / k
            d_hi = (verdict["dead"] + verdict["unknown"]) / k
            row.update(
                focal_status=r.status, focal_expanded=r.expanded, focal_sec=round(fsec, 2),
                n_sample=len(res), **verdict,
                dead_frac=round(d_lo, 4), dead_or_unknown_frac=round(d_hi, 4),
                ceiling_lo=round(1 / (1 - d_lo), 4) if d_lo < 1 else None,
                ceiling_hi=round(1 / (1 - d_hi), 4) if d_hi < 1 else None,
                # 막다른 상태가 "즉시" (후속 0 — 1 pop) 인가 "웅덩이" 인가
                dead_immediate=sum(1 for n in dead_pops if n <= 1),
                dead_pops_median=statistics.median(dead_pops) if dead_pops else None,
                judge_sec=round(jsec, 2))
            rows.append(row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 포컬 {r.status} exp {r.expanded} "
                  f"({fsec:.1f}s) | 표본 {len(res)} → DEAD {verdict['dead']} · UNK {verdict['unknown']} "
                  f"· ALIVE {verdict['alive']} = d {d_lo:.3f}~{d_hi:.3f} → 상한 ×{row['ceiling_lo']}"
                  f"~{row['ceiling_hi']} ({jsec:.1f}s)", flush=True)

    used = [r for r in rows if "dead_frac" in r]
    dl = [r["dead_frac"] for r in used]
    dh = [r["dead_or_unknown_frac"] for r in used]
    lo = [r["ceiling_lo"] for r in used if r["ceiling_lo"]]
    hi = [r["ceiling_hi"] for r in used if r["ceiling_hi"]]
    tot = {k: sum(r[k] for r in used) for k in ("dead", "unknown", "alive", "n_sample")}
    summ = {
        "n_routes": len(used), "eps": a.eps, "priority_mode": a.priority_mode,
        "pooled_dead_frac": round(tot["dead"] / max(tot["n_sample"], 1), 4),
        "pooled_dead_or_unknown_frac": round((tot["dead"] + tot["unknown"]) / max(tot["n_sample"], 1), 4),
        "ceiling_lo_median": round(statistics.median(lo), 4) if lo else None,
        "ceiling_hi_median": round(statistics.median(hi), 4) if hi else None,
        "ceiling_lo_max": max(lo) if lo else None,
        "gate": "상한 배율 중앙 < 1.2 이면 접는다 (측정 전 선언)",
        # ★ UNKNOWN 이 많을 때의 해석 (스모크 후, 본 측정 **전** 고정 — 2026-11-09).
        # DEAD 는 "도달 가능 상태를 전부 소진" 이라 증명이고, ALIVE 는 cap 안에 목표를 못 보면
        # UNKNOWN 이 된다. 그래서 통과는 하한(DEAD 만)으로, 폐기는 상한(DEAD+UNKNOWN)으로 가른다.
        # 배율 1/(1−d) 은 d 에 단조이므로 **d 의 중앙값**으로 가른다 — d = 1 (배율 ∞) 인 라우트를
        # 목록에서 빼면 폐기 쪽으로 치우친다.
        "dead_frac_median": round(statistics.median(dl), 4) if dl else None,
        "dead_or_unknown_frac_median": round(statistics.median(dh), 4) if dh else None,
        "verdict": ("pass" if dl and 1 / (1 - min(statistics.median(dl), 0.999999)) >= 1.2 else
                    "kill" if dh and 1 / (1 - min(statistics.median(dh), 0.999999)) < 1.2 else
                    "inconclusive"),
        "dead_immediate_share": round(sum(r["dead_immediate"] for r in used) / max(tot["dead"], 1), 4),
        **{f"total_{k}": v for k, v in tot.items()},
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"support_standard": std, "ckpt": a.ckpt, "seeds": [seeds[0], seeds[-1]],
                               "budget": a.budget, "k": a.k, "cap": a.cap,
                               "rows": rows, "summary": summ}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n요약: {json.dumps(summ, ensure_ascii=False)}\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
