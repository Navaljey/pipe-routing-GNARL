"""`focal_search_dual` 검증 — 이중 큐 구현이 보장을 지키는가.

세 가지를 본다.

1. **ε=0 등가성** — priority 를 넣어도 `ε=0` 이면 FOCAL = {f = f_min} 이므로 반환 비용이
   skeleton A\\* 최적과 **같아야** 한다. priority 는 동점 순서만 바꾼다.
2. **(1+ε) 보장** — `ε>0` 에서 반환 **라우터 비용** ≤ (1+ε) × 최적. 설계 근거가 실제로
   성립하는지 숫자로 확인한다 (D99: 보장은 근거, 판정은 측정 — 그래도 근거는 맞아야 한다).
3. **최악 priority** — 일부러 **나쁜** priority(무작위·역순)를 넣어도 ① 비용 보장이 깨지지
   않고 ② 확장 수가 늘어날 뿐임을 본다. §9 의 "1단계 실패 양식은 J 초과가 아니라
   느려짐" 을 코드로 확인하는 것이다.

    PIPE_SUPPORT_STANDARD=... python3 v3/test_focal_dual.py --limit 12
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.layered import (G0, LayeredGraph, _route,              # noqa: E402
                                  clear_cache, heuristic_factory)
from pipe_routing.scenario import load                                   # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import focal, standard_guard                                     # noqa: E402

BUDGET = 400_000      # 시험용 — 긴 라우트는 건너뛴다


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="manual")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--eps", type=float, default=0.01)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    standard_guard.resolve(a.allow_public_standard, what="이중 큐 시험")

    files = sorted((SK / "scenarios" / a.set).glob("*.json"))
    rng = random.Random(0)
    n = n_eq = n_bound = 0
    worst_ratio = 1.0
    exp_ratios = []
    for f in files:
        clear_cache()
        sc = load(str(f))
        for pipe in sc.pipes:
            if n >= a.limit:
                break
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            h = heuristic_factory(g, pipe)
            ref = _route(g, pipe, 1e9, BUDGET)
            if ref.status != "ok":
                continue
            n += 1
            # ① ε=0 + priority → 최적과 같아야 한다
            r0 = focal.focal_search_dual(g, pipe, h, eps=0.0,
                                         priority=lambda s, gv, hv, fv: rng.random(),
                                         expand_limit=BUDGET * 4)
            eq = r0.status == "ok" and abs(r0.cost - ref.J) < 1e-6
            n_eq += eq
            # ③ ε>0 + 나쁜 priority → 보장은 지켜지고 확장만 는다
            rb = focal.focal_search_dual(g, pipe, h, eps=a.eps,
                                         priority=lambda s, gv, hv, fv: -(gv + hv),
                                         expand_limit=BUDGET * 8)
            ok_bound = rb.status == "ok" and rb.cost <= ref.J * (1 + a.eps) + 1e-6
            n_bound += ok_bound
            if rb.status == "ok":
                worst_ratio = max(worst_ratio, rb.cost / ref.J)
                exp_ratios.append(rb.expanded / max(ref.expanded, 1))
            print(f"  {f.stem}/{pipe.id} {pipe.nominal_size}: "
                  f"ε=0 {'=' if eq else '≠'} 최적 ({r0.cost:.3f} vs {ref.J:.3f}) · "
                  f"나쁜 priority 비용비 {(rb.cost / ref.J) if rb.status == 'ok' else float('nan'):.5f} · "
                  f"확장 {ref.expanded} → {rb.expanded}", flush=True)
        if n >= a.limit:
            break

    print(f"\n라우트 {n} · ε=0 최적 일치 {n_eq}/{n} · (1+{a.eps}) 보장 {n_bound}/{n} · "
          f"나쁜 priority 최악 비용비 {worst_ratio:.5f}")
    if exp_ratios:
        exp_ratios.sort()
        print(f"나쁜 priority 확장 배수 중앙 {exp_ratios[len(exp_ratios) // 2]:.2f}배 "
              f"(최대 {exp_ratios[-1]:.2f}배) — 느려지지만 비용은 안 깨진다")
    return 0 if (n_eq == n and n_bound == n and n > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
