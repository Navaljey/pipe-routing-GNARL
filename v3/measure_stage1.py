"""1단계 측정 — 학습 `ĥ` 포컬 탐색 vs 기준선 A\\*. **검증 시드**로만 돌린다.

세 가지를 **분리해서** 낸다 (설계안 §15.3). 섞으면 "학습이 안 됐는지 구현이 느린지" 를
구분할 수 없다.

    ① 확장 감소 배수 R = E_기준 / E_V3        ← 학습 품질. 구현과 무관
    ② 구현 단가 c_s (확장당 탐색) · c_h (상태당 추론)
    ③ 벽시계 = ① × ② — **승리 조건은 이것**이다 (D98)

품질은 **라우터 비용**(ε 보장이 걸린 값)과 **검증기 J** 를 둘 다 낸다 (D99).
여기서는 라우터 비용까지 재고, 검증기 J 는 skeleton Session 으로 따로 채점한다.

기준선은 **같은 표현·같은 예산**의 skeleton 컴파일 A\\* 다 (`_route`, 2²²).
⚠️ **기준선 벽시계에 고정층 빌드(`build_sec`)를 포함한다** — 측정 중 발견: `_FIXED`
LRU 캐시 때문에 두 번째 빌드가 거의 공짜라 앞서 h 느슨함 측정의 1.1 s 중앙에는
빌드가 빠져 있었다. 우리 쪽도 같은 그래프를 쓰므로 **빌드는 양쪽 공통 비용**이고
`search_sec` 만 비교하는 것이 맞다. 둘 다 기록한다.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_stage1.py \\
        --ckpt ckpt/h_stage1.pt --seed0 40000000 --n 15 --eps 0.01 --out out/stage1_eps01.json
"""

from __future__ import annotations

import argparse
import json
import os
import math
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
from v3 import features, focal, guards, standard_guard                   # noqa: E402

GEN_KW = dict(n_pipes=10)


def load_net(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v.numpy() for k, v in ck["state_dict"].items()}
    return NumpyHNet(sd), ck


class NumpyHNet:
    """학습된 가중치를 numpy 로 옮긴 추론 경로.

    왜 torch 를 안 쓰는가 — torch 의 단일 상태 forward 는 호출 오버헤드가 약 100 µs 라
    **기준선 확장 1회(5.37 µs)의 20배**다 (§15.1). numpy 작은 행렬곱이면 수 µs 다.
    배포 형태(컴파일)에 더 가까운 단가를 재기 위한 것이기도 하다 — 벽시계 판정에서
    "구현이 느린 것" 과 "학습이 안 된 것" 을 섞지 않으려면 추론 경로가 현실적이어야 한다.
    """

    def __init__(self, sd: dict):
        self.W: list[tuple[np.ndarray, np.ndarray]] = []
        i = 0
        while f"trunk.{i}.weight" in sd:
            self.W.append((np.asarray(sd[f"trunk.{i}.weight"], dtype=np.float32).T,
                           np.asarray(sd[f"trunk.{i}.bias"], dtype=np.float32)))
            i += 2                      # Linear, SiLU 번갈아
        self.wp = np.asarray(sd["head_pipe.weight"], dtype=np.float32).ravel()
        self.bp = float(np.asarray(sd["head_pipe.bias"]).ravel()[0])
        self.ws = np.asarray(sd["head_sup.weight"], dtype=np.float32).ravel()
        self.bs = float(np.asarray(sd["head_sup.bias"]).ravel()[0])
        self.has_d = "head_dist.weight" in sd
        if self.has_d:
            self.wd = np.asarray(sd["head_dist.weight"], dtype=np.float32).ravel()
            self.bd = float(np.asarray(sd["head_dist.bias"]).ravel()[0])

    @staticmethod
    def _softplus(x: float) -> float:
        # 수치 안정 — 큰 |x| 에서 exp 가 넘치지 않게 분기한다
        return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))

    def __call__(self, x: np.ndarray, h_val: float) -> tuple[float, float]:
        """→ (ĥ = 남은 비용 kg, d̂ = 남은 홉 수). d̂ 머리가 없으면 d̂ = nan."""
        z = x
        for W, b in self.W:
            z = z @ W + b
            np.clip(z, -30.0, 30.0, out=z)
            z = z / (1.0 + np.exp(-z))          # SiLU
        hh = (h_val + self._softplus(float(z @ self.wp) + self.bp)
              + self._softplus(float(z @ self.ws) + self.bs))
        dd = self._softplus(float(z @ self.wd) + self.bd) if self.has_d else float("nan")
        return hh, dd


def make_priority(net, ck, g, pipe, h, counter, mode: str = "d_only"):
    """priority(state, g, h, f) → 작은 값을 먼저. 상태당 1회 호출이므로 비용 = c_h 다.

    **왜 `d_only` 가 기본인가 — 파일럿에서 배운 것.**
    `ĥ`(kg)를 priority 로 쓴 1차 측정의 확장 감소가 R = 1.06~1.92 (중앙 약 1.4) 였다.
    `ĥ` 는 **비용** 추정이고 확장 수를 줄이는 것은 **거리**다. 유계 준최적 탐색에서
    품질은 OPEN 의 admissible `h` 와 ε 가 지키고, FOCAL 안 순서는 **목표까지 남은 홉 수**
    `d̂` 로 잡는 것이 정석이다 (Speedy search · Explicit Estimation Search 의 h vs d).
    ε 보장은 OPEN 쪽에 걸려 있으므로 priority 를 바꿔도 **깨지지 않는다.**

    ``d_only``    `d̂` = 남은 홉 수. 속도용 (기본)
    ``h_only``    `ĥ` = 남은 비용. 1차 파일럿이 쓴 것 — 대조군으로 남긴다
    ``g_plus_h``  `g + ĥ` = 추정 전체 비용. 가장 보수적
    ``d_then_h``  `d̂` 우선, 동점이면 `ĥ` — 홉 수가 같을 때만 비용으로 가른다
    """
    fm = features.FeatureMaker(g, pipe)
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)

    def priority(s, gv, hv, f):
        counter[0] += 1
        hh, dd = net((fm.one(s, hv) - mu) / sd, hv)
        if mode == "d_only":
            return dd
        if mode == "h_only":
            return hh
        if mode == "g_plus_h":
            return gv + hh
        return (round(dd, 3), hh)        # d_then_h

    return priority


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="1단계 측정 (검증 시드)")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01])
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--v3-budget", type=int, default=0,
                    help="V3 쪽 확장 예산 (0 = 기준선과 같게). D97 자기 결정적 예산")
    ap.add_argument("--out", required=True)
    ap.add_argument("--priority-mode", default="d_only",
                    choices=["d_only", "h_only", "g_plus_h", "d_then_h", "oracle"])
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")          # ★ 평가 시드(20260000~) 로 못 돌린다
    std = standard_guard.resolve(a.allow_public_standard, what="1단계 측정")
    net, ck = (load_net(a.ckpt) if a.ckpt else (None, {}))
    v3_budget = a.v3_budget or a.budget

    rows: list[dict] = []
    for seed in seeds:
        name = f"valid_{seed}"
        try:
            sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        except Exception as e:
            print(f"  {name}: 생성 실패 {e}", flush=True)
            continue
        for pipe in sc.pipes:
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            hf = heuristic_factory(g, pipe)
            row = {"scenario": name, "seed": seed, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe), "build_sec": round(g.build_sec, 3)}
            # ── 기준선: skeleton 컴파일 A* (같은 그래프·같은 예산)
            rb = _route(g, pipe, 1e9, a.budget)
            row["base"] = {"status": rb.status, "expanded": rb.expanded,
                           "search_sec": round(rb.search_sec, 4), "cost": rb.J,
                           "J_support_est": rb.J_support_est}
            # ── V3: 이중 큐 포컬 + 학습 ĥ (또는 오라클)
            oracle = None
            if a.priority_mode == "oracle":
                # ★ **상한 측정.** 기준선 최적 경로 위 상태를 0, 나머지를 1 로 둔다.
                # 어떤 priority 함수도 이보다 잘할 수 없다 (경로를 미리 아는 셈이므로).
                # 이것이 "순위가 지렛대인가" 를 가르는 측정이다 — 오라클 R 이 작으면
                # 띠 안 순서에는 애초에 여지가 없고, 1단계 속도 이야기가 성립하지 않는다.
                if rb.status != "ok":
                    rows.append(row)
                    continue
                oracle = {(int(s.node), int(s.dir), round(float(s.run), 3),
                           int(s.bend)) for s in rb.states}
            for eps in a.eps:
                cnt = [0]
                if oracle is not None:
                    def pr(s, gv, hv, f, _o=oracle, _c=cnt):
                        _c[0] += 1
                        return 0.0 if (int(s.node), int(s.dir), round(float(s.run), 3),
                                       int(s.bend)) in _o else 1.0
                else:
                    pr = make_priority(net, ck, g, pipe, hf, cnt, a.priority_mode)
                t0 = time.perf_counter()
                r = focal.focal_search_dual(g, pipe, hf, eps=eps, priority=pr,
                                            expand_limit=v3_budget)
                sec = time.perf_counter() - t0
                row[f"eps_{eps}"] = {"status": r.status, "expanded": r.expanded,
                                     "search_sec": round(sec, 4), "cost": r.cost,
                                     "n_priority_calls": cnt[0],
                                     "c_h_us": round(sec / cnt[0] * 1e6, 2) if cnt[0] else None,
                                     "ratio_cost": (r.cost / rb.J) if (r.cost and rb.J) else None,
                                     "ratio_expanded": (rb.expanded / r.expanded) if r.expanded else None}
            rows.append(row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 기준 {rb.status} "
                  f"exp {rb.expanded} {rb.search_sec:.2f}s | "
                  + " | ".join(f"ε={e} {row[f'eps_{e}']['status']} exp {row[f'eps_{e}']['expanded']} "
                               f"{row[f'eps_{e}']['search_sec']:.2f}s" for e in a.eps), flush=True)

    out = {"support_standard": std, "ckpt": a.ckpt, "seeds": [seeds[0], seeds[-1]],
           "budget": a.budget, "v3_budget": v3_budget, "eps": a.eps,
           "priority_mode": a.priority_mode,
           "flops_per_state": ck.get("flops_per_state"), "rows": rows}
    # 요약 — 양쪽 모두 ok 인 라우트에서만 짝지어 본다
    for eps in a.eps:
        k = f"eps_{eps}"
        pair = [r for r in rows if r["base"]["status"] == "ok" and r[k]["status"] == "ok"]
        summ = {"n_paired": len(pair),
                "base_ok": sum(r["base"]["status"] == "ok" for r in rows),
                "v3_ok": sum(r[k]["status"] == "ok" for r in rows), "n_routes": len(rows)}
        if pair:
            summ.update(
                R_expanded_median=round(statistics.median(r[k]["ratio_expanded"] for r in pair), 3),
                cost_ratio_median=round(statistics.median(r[k]["ratio_cost"] for r in pair), 5),
                cost_ratio_max=round(max(r[k]["ratio_cost"] for r in pair), 5),
                sec_base_sum=round(sum(r["base"]["search_sec"] for r in pair), 2),
                sec_v3_sum=round(sum(r[k]["search_sec"] for r in pair), 2),
                c_h_us_median=round(statistics.median(
                    r[k]["c_h_us"] for r in pair if r[k]["c_h_us"]), 2))
        out[f"summary_{eps}"] = summ
        print(f"\nε={eps}: {json.dumps(summ, ensure_ascii=False)}")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
