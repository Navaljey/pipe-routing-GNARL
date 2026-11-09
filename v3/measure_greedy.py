"""망 단독 보행 측정 — **"신경망이 A\\* 를 대체한다" 를 직접 판정한다** (2026-11-05 소유자 지시).

**왜 이 측정이 필요한가.** 설계안 §6·§19 는 *"망이 경로를 직접 내면 안 된다"* 를
전제로 깔고 있는데, 그 근거인 §19 의 붕괴(0.271 → 0.000)와 §16 의 천장(0.40~0.45)은
**다른 망**에서 나왔다 — 강화학습 · h 드리프트 1.32 → 13 kg · τ 붕괴 0.05 → 0.0117 ·
격자 20×20×10. 지금 있는 망은 **지도학습**이고 드리프트·τ 기전이 없다. 즉
**오늘의 망으로는 아무도 재보지 않았다.** 그래서 전제를 측정으로 교체한다.

재는 것 — **도착하느냐**다. 속도 산수는 오히려 보행 쪽에 유리하다:

    보행   경로 300칸 × 이웃 18 = 5,400 상태 × 21 µs  ≈ 0.11 s
    기준선 확장 272,922 × 5.37 µs                     ≈ 1.47 s   → 13배

막는 것은 D77 ① 이다. E1 은 시나리오당 required 약 17.8 본이고 **전부 성공**해야
완전 성공 1건이므로 배관당 도달률이 거듭제곱된다 — 0.45^17.8 ≈ 0.0007 (0/41),
기준선 7/41 을 내려면 배관당 **90.6%** 가 필요하다. 그래서 1차 지표가 도달률이다.

**두 모드 = Jay 의 목표와 현재 설계 사이의 연속선.**

    beam = 1   argmin(ΔJ + ĥ) 한 칸씩. OPEN 없음 · 되돌아감 없음 = 순수 보행
    beam = B   부분 경로 B개를 f̂ = g + ĥ 로 유지 = 되돌아감을 B 만큼만 허용
    (B → ∞ 이면 포컬 탐색에 수렴한다. 그 끝이 지금의 설계다)

⚠️ **순환 회피는 되돌아감이 아니다.** 방문한 상태를 후보에서 빼지 않으면 보행이
제자리에서 영원히 돈다(종료 자체가 안 된다). 종료를 위한 최소 장치이고,
**막혔을 때 앞 갈림길로 돌아가지는 않는다** — 그것이 되돌아감이다.

Layer 0 마스크는 유지한다. `space.neighbors()` 가 이미 적용하므로 보행이 내는 경로는
**불법이 아니다** — 못 가거나(데드락), 가더라도 J 가 나쁠 뿐이다. 그 둘을 가른다.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_greedy.py \\
        --ckpt ckpt/h_pilot_h32.pt --seed0 40000000 --n 2 --beam 1 4 \\
        --out out/greedy.json
"""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import features, guards, standard_guard                          # noqa: E402
from v3.measure_stage1 import load_net                                   # noqa: E402

GEN_KW = dict(n_pipes=10)


def greedy_walk(space, pipe, hf, hhat, *, beam: int = 1, step_limit: int = 20_000):
    """f̂ = g + ĥ 로 부분 경로 `beam` 개를 유지하는 보행. beam=1 이면 순수 탐욕.

    → (status, g_final, steps, n_states, h_frac_at_fail)

    `h_frac_at_fail` = 막힌 지점의 admissible h ÷ 출발점 h. **어디서 막혔나**의 척도다 —
    1.0 이면 출발 직후, 0 에 가까우면 목표 문앞에서 막혔다는 뜻이고 처방이 다르다.
    """
    start = space.start_state(pipe)
    h0 = max(hf(start), 1e-9)
    # 각 beam 항목 = (f̂, g, state). 방문 집합은 **beam 전체 공통**이다 —
    # 따로 두면 B개가 같은 길을 중복해서 걸어 beam 이 사실상 1 이 된다.
    frontier = [(hf(start) + 0.0, 0.0, start)]
    seen = {start}
    n_states = 0
    best_h_frac = 1.0

    for step in range(1, step_limit + 1):
        cand: list[tuple[float, float, object]] = []
        for _f, g, s in frontier:
            for nb in space.neighbors(s, pipe):
                if nb in seen:
                    continue
                ng = g + space.cost(s, nb, pipe)
                if space.is_goal(nb, pipe):
                    return "ok", ng, step, n_states, 0.0
                hv = hf(nb)
                best_h_frac = min(best_h_frac, hv / h0)
                n_states += 1
                cand.append((ng + hhat(nb, hv), ng, nb))
        if not cand:
            # 열린 이웃이 하나도 없다 = 데드락. 되돌아갈 장치가 없으므로 여기서 끝이다
            return "deadlock", None, step, n_states, best_h_frac
        cand.sort(key=lambda t: t[0])
        frontier = cand[:beam]
        seen.update(s for _f, _g, s in frontier)

    return "step_limit", None, step_limit, n_states, best_h_frac


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="망 단독 보행 측정 (검증 시드)")
    ap.add_argument("--ckpt", required=True, help="ĥ 체크포인트 (2머리)")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--step-limit", type=int, default=20_000)
    ap.add_argument("--out", required=True)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")      # ★ 평가 시드(20260000~) 로 못 돌린다
    std = standard_guard.resolve(a.allow_public_standard, what="망 단독 보행 측정")
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
            row = {"scenario": name, "seed": seed, "pipe_id": pipe.id,
                   "size": pipe.nominal_size, "gravity": bool(pipe.gravity_pipe),
                   "build_sec": round(g.build_sec, 3)}
            rb = _route(g, pipe, 1e9, a.budget)
            row["base"] = {"status": rb.status, "expanded": rb.expanded,
                           "search_sec": round(rb.search_sec, 4), "cost": rb.J}

            fm = features.FeatureMaker(g, pipe)
            cnt = [0, 0.0]

            def hhat(s, hv, _c=cnt):
                t = time.perf_counter()
                _c[0] += 1
                hh, _dd = net((fm.one(s, hv) - mu) / sd, hv)
                _c[1] += time.perf_counter() - t
                return hh

            parts = []
            for beam in a.beam:
                cnt[0] = 0
                cnt[1] = 0.0
                t0 = time.perf_counter()
                status, gf, steps, n_states, hfrac = greedy_walk(
                    g, pipe, hf, hhat, beam=beam, step_limit=a.step_limit)
                sec = time.perf_counter() - t0
                d = {"status": status, "steps": steps, "n_states": n_states,
                     "walk_sec": round(sec, 4), "cost": gf,
                     "c_h_us": round(cnt[1] / cnt[0] * 1e6, 2) if cnt[0] else 0.0,
                     "h_frac_at_fail": round(hfrac, 4)}
                if status == "ok" and rb.status == "ok" and rb.J:
                    d["ratio_cost"] = gf / rb.J
                    d["ratio_sec"] = sec / max(rb.search_sec, 1e-9)
                row[f"beam_{beam}"] = d
                parts.append(f"beam{beam} {status} {steps}칸 {sec:.2f}s"
                             + (f" J비 {d['ratio_cost']:.3f}" if "ratio_cost" in d else ""))
            rows.append(row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 기준 {rb.status} "
                  f"exp {rb.expanded} {rb.search_sec:.2f}s | " + " | ".join(parts), flush=True)

    out = {"support_standard": std, "ckpt": a.ckpt, "seeds": seeds,
           "budget": a.budget, "step_limit": a.step_limit, "beam": a.beam, "rows": rows}
    for beam in a.beam:
        k = f"beam_{beam}"
        # 분모는 **기준선이 푼 라우트**다 — 도달 불가 라우트를 분모에 넣으면
        # 보행의 도달률이 기준선의 한계 덕에 좋아 보인다
        solvable = [r for r in rows if r["base"]["status"] == "ok"]
        ok = [r for r in solvable if r[k]["status"] == "ok"]
        js = [r[k]["ratio_cost"] for r in ok if "ratio_cost" in r[k]]
        fails = [r for r in solvable if r[k]["status"] != "ok"]
        s = {"n_solvable": len(solvable), "n_ok": len(ok),
             "reach_rate": round(len(ok) / len(solvable), 4) if solvable else None,
             "n_deadlock": sum(1 for r in fails if r[k]["status"] == "deadlock"),
             "n_step_limit": sum(1 for r in fails if r[k]["status"] == "step_limit"),
             "J_ratio_median": round(statistics.median(js), 5) if js else None,
             "J_ratio_max": round(max(js), 5) if js else None,
             "walk_sec_median": round(statistics.median(
                 [r[k]["walk_sec"] for r in ok] or [0]), 3),
             "base_sec_median": round(statistics.median(
                 [r["base"]["search_sec"] for r in ok] or [0]), 3),
             "c_h_us_median": round(statistics.median(
                 [r[k]["c_h_us"] for r in solvable if r[k]["c_h_us"]] or [0]), 2),
             "h_frac_at_fail_median": round(statistics.median(
                 [r[k]["h_frac_at_fail"] for r in fails] or [0]), 4)}
        # 완전 성공 추정 — D77 ① 이 1순위이므로 배관당 도달률을 시나리오로 환산한다
        if s["reach_rate"] is not None:
            per_sc = {}
            for r in solvable:
                per_sc.setdefault(r["scenario"], []).append(r[k]["status"] == "ok")
            s["n_scenarios_all_ok"] = sum(1 for v in per_sc.values() if all(v))
            s["n_scenarios"] = len(per_sc)
        out[f"summary_beam_{beam}"] = s
        print(f"\nbeam={beam}: {json.dumps(s, ensure_ascii=False)}", flush=True)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
