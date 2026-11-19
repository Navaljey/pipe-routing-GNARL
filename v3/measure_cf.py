"""C안 벽시계 측정 — 컴파일 A* + 맵당 1회 노드 표 (설계안 §33, 2026-11-19).

변형 (같은 프로세스·같은 그래프·같은 예산 2²²):
  base      skeleton 컴파일 `_route` — 기준
  rg1.0     컴파일 + max(h, rg)              (D125 판정 상대)
  rg1.1     컴파일 + 1.1·max(h, rg)          (같은 품질대의 무료 근사, §30.8)
  cf_<ck>_<dec>  컴파일 + max(h, rg·CF̂)      CF̂ = 노드 표 (맵당 1회 배치 추론)

벽시계 = 탐색 + 사전 계산. rg 변형의 사전 계산 = 역방향 Dijkstra. cf 변형 = Dijkstra + 노드 특징 + MLP.
그래프 생성 시간은 공통이라 따로 적는다.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_cf.py --ckpt ckpt/cf_h64.npz --out out/cf_v1.json --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.astar_rg import route_rg                                         # noqa: E402
from v3.measure_cea import GEN_KW, verifier_J                            # noqa: E402
from v3.measure_controls import relaxed_dist                             # noqa: E402
from v3.measure_frontier_rg import _st, _verify_wp, summarize            # noqa: E402
from v3.node_table import NodeCF                                         # noqa: E402

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import EXPAND_BUDGET, G0, LayeredGraph, _route, clear_cache  # noqa: E402
from pipe_routing.router_astar import _finish_chain                     # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    ap.add_argument("--ckpt", nargs="+", required=True, help="이름=경로 또는 경로 (이름은 파일 stem)")
    ap.add_argument("--decode", default="q0.05,argmax")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")
    std_info = standard_guard.resolve(a.allow_public_standard, what="C안 측정")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")
    models = {}
    for c in a.ckpt:
        nm, path = c.split("=", 1) if "=" in c else (pathlib.Path(c).stem, c)
        for dec in a.decode.split(","):
            models[f"cf_{nm}_{dec}"] = NodeCF(path, dec)

    part = pathlib.Path(a.out + ".partial.jsonl")
    rows = []
    if a.resume and part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {(r["scenario"], r["pipe_id"]) for r in rows}
    warmed = False
    for seed in seeds:
        name = f"valid_{seed}"
        clear_cache()
        gc.collect()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            if (name, pipe.id) in done:
                continue
            t = time.perf_counter()
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            row = {"scenario": name, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe), "graph_sec": round(time.perf_counter() - t, 3), "v": {}}
            if not warmed:
                _route(g, pipe, 1e9, 2048)
                route_rg(g, pipe, np.zeros(len(g.pos)), 2048, 1.3)
                warmed = True
            t = time.perf_counter()
            rb = _route(g, pipe, 1e9, a.budget)
            v = {"status": _st(rb.status), "expanded": rb.expanded, "sec": round(time.perf_counter() - t, 4),
                 "pre_sec": 0.0}
            if v["status"] == "ok":
                v["cost"] = rb.J
                J, ok, _ = _verify_wp(sc, pipe, rb, std)
                v.update(J_verifier=J, verifier_success=ok)
            row["v"]["base"] = v
            t = time.perf_counter()
            dist, _ = relaxed_dist(g, pipe)
            pre_rg = time.perf_counter() - t
            s0 = g.start_state(pipe).node
            tabs = {"rg1.0": (dist, 1.0, pre_rg, None), "rg1.1": (dist, 1.1, pre_rg, None)}
            for key, m in models.items():
                t = time.perf_counter()
                htab, stat = m.table(g, pipe, dist, s0)
                tabs[key] = (htab, 1.0, pre_rg + (time.perf_counter() - t), stat)
            for key, (tab, w, pre, stat) in tabs.items():
                t = time.perf_counter()
                st, chain, ex, gen = route_rg(g, pipe, tab, a.budget, w)
                v = {"status": _st(st), "expanded": ex, "sec": round(time.perf_counter() - t, 4),
                     "pre_sec": round(pre, 4)}
                if stat:
                    v["table"] = stat
                if v["status"] == "ok":
                    v["cost"] = _finish_chain(g, pipe, chain, ex, gen, 0.0).J
                    J, ok, _ = verifier_J(sc, g, pipe, chain, std)
                    v.update(J_verifier=J, verifier_success=ok)
                row["v"][key] = v
            rows.append(row)
            with part.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: " + " | ".join(
                f"{k} {x['status'][:6]} {x['expanded']} {x['sec'] + x['pre_sec']:.2f}s" for k, x in row["v"].items()),
                flush=True)
            del g
        del sc
    summ = summarize(rows)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "budget": a.budget,
                                               "seeds": [seeds[0], seeds[-1]], "summary": summ, "rows": rows},
                                              ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
