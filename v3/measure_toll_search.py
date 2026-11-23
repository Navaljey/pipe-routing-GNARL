"""통행료 계수 탐색 — 2번 (설계안 §37.3, 측정 전 선언).

학습 시드 30100000~03 (밀집 30 구성, 4장) 에서 계수 18 점 (C · S · 무작위 16, 시드 고정) 을 같은 협상 (충돌 배관만,
rg × 1.1, 반복 ≤ 15, 확정, 검증기 채점) 으로 돌린다. 반복 1 (단독) 은 계수와 무관 → 맵마다 한 번만.
선택 = D77 사전식 (맵 4장 합): ① 완전 성공 ② 필수 통과 ③ 통과 배관 J 합. 검증은 measure_negotiate --variants P --params.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_toll_search.py --seed0 30100000 --n 1 --out out/toll_search_v1.json --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.measure_negotiate import (GEN_KW, Clock, Negotiator, commit,     # noqa: E402
                                  negotiate, prune_item_cache, rss_gb, score, variant_params)

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import EXPAND_BUDGET, clear_cache             # noqa: E402

N_RANDOM = 16
RNG_SEED = 20261123


def configs() -> list[dict]:
    rng = np.random.default_rng(RNG_SEED)
    out = [dict(name="C", **variant_params("C")), dict(name="S", **variant_params("S"))]
    for i in range(N_RANDOM):
        out.append(dict(name=f"r{i:02d}",
                        pres0=round(float(10 ** rng.uniform(math.log10(0.1), math.log10(2.0))), 4),
                        pres_mul=round(float(rng.uniform(1.1, 2.5)), 4),
                        hist_add=round(float(rng.uniform(0.0, 1.0)), 4),
                        alpha=round(float(rng.uniform(0.0, 1.0)), 4),
                        s=round(float(rng.uniform(0.0, 3.0)), 4),
                        support=True))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="통행료 계수 탐색 (§37.3)")
    ap.add_argument("--seed0", type=int, default=30_100_000)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "train")
    std_info = standard_guard.resolve(a.allow_public_standard, what="통행료 계수 탐색")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")
    part = pathlib.Path(a.out + ".partial.jsonl")
    rows = []
    if a.resume and part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {(r["scenario"], r["config"]) for r in rows}
    cfgs = configs()
    for seed in seeds:
        name = f"train_{seed}"
        if all((name, c["name"]) in done for c in cfgs):
            continue
        clear_cache()
        gc.collect()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        neg = Negotiator(sc, a.budget)
        clk1, calls1 = Clock(), []
        t0 = time.perf_counter()
        paths0, failed0 = {}, set()
        for pid in neg.order:
            st, wp = neg.route(pid, {}, lambda x: 0.0, clk1, calls1, 1)
            paths0[pid] = wp
            if st != "ok":
                failed0.add(pid)
        wall1 = time.perf_counter() - t0
        print(f"  {name} 반복 1: {wall1:.0f} s · 실패 {len(failed0)}", flush=True)
        for c in cfgs:
            if (name, c["name"]) in done:
                continue
            neg.p = {k: c[k] for k in ("pres0", "pres_mul", "hist_add", "alpha", "s", "support")}
            neg.toll = "self"
            clk, calls = Clock(clk1), list(calls1)
            t = time.perf_counter()
            paths, conf, trace = negotiate(neg, paths0, set(failed0), "C", clk, calls, t - wall1, wall1)
            final, n_commit = commit(neg, paths, conf, clk, calls)
            sc_out = score(sc, final, std)
            row = {"scenario": name, "config": c["name"], "params": neg.p, "iterations": trace[-1]["it"],
                   "converged": not conf, "commit_rerouted": n_commit, "calls": len(calls),
                   "wall_sec": round(wall1 + time.perf_counter() - t, 2), "trace": trace, "score": sc_out,
                   "support_standard": std_info.get("sha256")}
            with part.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            rows.append(row)
            print(f"  {name} {c['name']}: 반복 {row['iterations']} · 수렴 {row['converged']} · 필수 "
                  f"{sc_out['required_passed']}/{sc_out['required_pipes']} · J {sc_out['J_total']:.0f} · "
                  f"{row['wall_sec']:.0f} s · rss {rss_gb():.1f} GB", flush=True)
            prune_item_cache(neg, {})
        del sc, neg
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "configs": cfgs, "rows": rows},
                                              ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


def select(rows, cfg_names=None) -> list:
    """D77 사전식 순위 (맵 합): ① 완전 성공 ② 필수 통과 ③ −J."""
    by = {}
    for r in rows:
        b = by.setdefault(r["config"], {"full": 0, "passed": 0, "J": 0.0, "maps": 0})
        b["full"] += int(r["score"]["full_success"])
        b["passed"] += r["score"]["required_passed"]
        b["J"] += r["score"]["J_total"]
        b["maps"] += 1
    n = max(b["maps"] for b in by.values())
    ok = {k: v for k, v in by.items() if v["maps"] == n}
    return sorted(ok.items(), key=lambda kv: (-kv[1]["full"], -kv[1]["passed"], kv[1]["J"]))


if __name__ == "__main__":
    raise SystemExit(main())
