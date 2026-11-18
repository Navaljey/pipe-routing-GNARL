"""무료 대조군 vs CEA — 같은 파이썬 A\\* 에서 `f` 만 바꿔 R·품질을 잰다 (설계안 §30.2).

    base        f = g + h                       최적
    wa<w>       f = g + w·h                     ≤ w·C*  (가중 A*, 추론 비용 0)
    rg          f = g + max(h, h_RG)            최적    (h_RG = 완화 그래프 정확 거리, 아래)
    cea_<dec>   f = g + max(h, h·ê)             Theorem 3

**h_RG (완화 그래프 거리).** 노드 그래프(`step`·`elen`·`scost`·`emask`)에서 방향·직진 길이·꺾임·`su`·구배를
전부 무시하고, 엣지 무게 = 길이 × kg/m + 엣지 추정 서포트(`scost`) 로 **목표 노드까지** 역방향 Dijkstra.
실제 엣지 비용 = 길이 × kg/m + 엘보 + `scost` 이고 엘보 ≥ 0 이므로 엣지 무게 ≤ 실제 비용 →
완화 그래프 거리 ≤ 실제 남은 비용 (admissible), 삼각 부등식으로 consistent. **`h` 가 못 보는 서포트를 본다.**
ALT 에서 랜드마크 = 목표 하나인 경우와 같다 (그때 하한이 가장 단단하다). 사전 계산 시간은 따로 재고
벽시계에 더한다.

    PIPE_SUPPORT_STANDARD=... python3 v3/measure_controls.py --seed0 40000000 --n 10 \\
        --variants base,wa1.1,wa1.2,wa1.5,rg,cea_q0.05,cea_argmax --ckpt ckpt/cea_1b.npz \\
        --out out/controls.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.measure_cea import (GEN_KW, astar_py, make_cea_h, rss_gb,         # noqa: E402
                            verifier_J)
from v3.train_cea import CEAHeuristic                                    # noqa: E402

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,       # noqa: E402
                                  _route, clear_cache, heuristic_factory)
from pipe_routing.pipe_spec import pipe_spec                            # noqa: E402


def relaxed_dist(g, pipe) -> tuple[np.ndarray, float]:
    """목표 노드까지 완화 그래프 거리 (kg). → (노드별 거리, 계산 초)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    t0 = time.perf_counter()
    kgpm = pipe_spec(pipe).kg_per_m       # S2-0b — 배관별 사양은 이 함수로만
    n = g.step.shape[0]
    src, d = np.nonzero((g.step >= 0) & (g.emask != 0))
    dst = g.step[src, d]
    w = g.elen[src, d] / 1000.0 * kgpm + g.scost[src, d]
    # 같은 (src, dst) 쌍이 여럿이면 가장 싼 것만 — coo → csr 은 중복을 **더하므로** 먼저 줄인다.
    # escape graph 는 (노드, 방향) 마다 이웃이 하나라 보통 중복이 없다 → 확인만 하고 있을 때만 정렬 (결과 같음)
    key = src.astype(np.int64) * n + dst
    if len(np.unique(key)) != len(key):
        order = np.lexsort((w, key))
        key, w, src, dst = key[order], w[order], src[order], dst[order]
        first = np.ones(len(key), bool)
        first[1:] = key[1:] != key[:-1]
        src, dst, w = src[first], dst[first], w[first]
    w = np.maximum(w, 1e-12)            # 0 무게는 csgraph 가 '엣지 없음' 으로 읽는다
    # 역방향: dst → src 로 뒤집어 목표에서 한 번에
    m = coo_matrix((w, (dst, src)), shape=(n, n)).tocsr()
    goal = _goal_node(g, pipe)
    dist = dijkstra(m, directed=True, indices=goal)
    return dist, time.perf_counter() - t0


def _goal_node(g, pipe) -> int:
    end = np.asarray(pipe.end.pos, float)
    k = int(np.argmin(np.abs(g.pos - end).sum(1)))
    assert np.abs(g.pos[k] - end).sum() < 1e-6, "목표 노드를 못 찾았다"
    return k


def make_h(var: str, g, pipe, hf, cea_cache: dict, ckpt: str, stat: dict):
    if var == "base":
        return hf
    if var.startswith("wa"):
        w = float(var[2:])
        return lambda s: w * hf(s)
    if var == "rg":
        dist, sec = relaxed_dist(g, pipe)
        stat["pre_sec"] = sec
        stat["unreach_nodes"] = int(np.isinf(dist).sum())

        def h(s):
            v = dist[s.node]
            hv = hf(s)
            return hv if v < hv else float(v)      # inf 면 그대로 inf → 그 상태는 사실상 버려진다
        return h
    if var.startswith("cearg"):
        # §30.10 ②-a: rg 위 CEA — 변형 이름 cearg<은닉>_<디코딩>, 체크포인트 ckpt/cea_rg_h<은닉>.npz
        hid, dec = var[5:].split("_", 1)
        key = (hid, dec)
        if key not in cea_cache:
            cea_cache[key] = CEAHeuristic(f"ckpt/cea_rg_h{hid}.npz", dec)
        dist, sec = relaxed_dist(g, pipe)
        stat["pre_sec"] = sec
        return make_cea_rg_h(cea_cache[key], g, pipe, hf, dist, stat)
    if var.startswith("cea_"):
        dec = var[4:]
        if dec not in cea_cache:
            cea_cache[dec] = CEAHeuristic(ckpt, dec)
        return make_cea_h(cea_cache[dec], g, pipe, hf, stat)
    raise SystemExit(f"모르는 변형 {var}")


def make_cea_rg_h(cea, g, pipe, hf, dist, stat):
    """max(hb, hb·ê), hb = max(h, rg). 입력 = 기존 특징 + [rg_norm, log(hb/h)] (train_cea --base-h rg 와 같은 식)."""
    from v3 import features
    assert str(getattr(cea, "base_h", "rg")) in ("rg",)
    fm = features.FeatureMaker(g, pipe)
    ih = features.NAMES.index("h_norm")
    cache: dict = {}

    def h(s):
        v = cache.get(s)
        if v is not None:
            return v
        hv = hf(s)
        rgv = float(dist[s.node])
        hb = rgv if rgv > hv else hv
        if hv <= 0.0 or not np.isfinite(hb):
            v = hb
        else:
            t = time.perf_counter()
            x = fm.one(s, hv)
            scale = float(x[ih]) / hv
            xe = np.concatenate([x, np.asarray([rgv * scale, math.log(hb / hv)], np.float32)])
            r = cea.ratio(xe)
            stat["sec"] += time.perf_counter() - t
            stat["calls"] += 1
            stat["lifted"] += r > 1.0
            v = hb * r if r > 1.0 else hb
        cache[s] = v
        return v
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="무료 대조군 vs CEA (설계안 §30.2)")
    ap.add_argument("--variants", default="base,wa1.1,wa1.2,wa1.5,rg,cea_q0.05,cea_argmax")
    ap.add_argument("--ckpt", default="ckpt/cea_1b.npz")
    ap.add_argument("--seed0", type=int, default=40_000_000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--py-budget", type=int, default=600_000)
    ap.add_argument("--py-time", type=float, default=180.0)
    ap.add_argument("--mem-limit", type=float, default=3.5, help="탐색 1회 RSS 증가분 상한 (GB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    variants = a.variants.split(",")
    assert variants[0] == "base", "첫 변형은 base (짝의 기준)"
    seeds = [a.seed0 + i for i in range(a.n)]
    guards.assert_seed_range(seeds, "valid")          # ★ 평가 시드 금지
    std_info = standard_guard.resolve(a.allow_public_standard, what="대조군 측정")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")

    part = pathlib.Path(a.out + ".partial.jsonl")
    rows, done = [], set()
    if a.resume and part.exists():
        for line in part.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if (r["scenario"], r["pipe_id"]) not in done:
                done.add((r["scenario"], r["pipe_id"]))
                rows.append(r)
        print(f"이어서: {len(done)} 배관", flush=True)
    elif part.exists():
        raise SystemExit(f"{part} 가 이미 있다 — --resume")
    cea_cache: dict = {}

    for seed in seeds:
        name = f"valid_{seed}"
        clear_cache()                    # ★ 고정층 캐시 — 시나리오 사이 메모리 누적의 원인 후보
        gc.collect()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            if (name, pipe.id) in done:
                continue
            gc.collect()
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            hf = heuristic_factory(g, pipe)
            row = {"scenario": name, "seed": seed, "pipe_id": pipe.id, "size": pipe.nominal_size,
                   "gravity": bool(pipe.gravity_pipe)}
            t = time.perf_counter()
            rb = _route(g, pipe, 1e9, EXPAND_BUDGET)
            row["compiled"] = {"status": rb.status, "expanded": rb.expanded, "cost": rb.J,
                               "sec": round(time.perf_counter() - t, 3)}
            if rb.status != "unreachable":
                for var in variants:
                    stat = {"sec": 0.0, "calls": 0, "lifted": 0}
                    h = make_h(var, g, pipe, hf, cea_cache, a.ckpt, stat)
                    st, cost, chain, ex, gen, reo, sec = astar_py(
                        g, pipe, h, expand_limit=a.py_budget, time_limit=a.py_time, mem_limit=a.mem_limit)
                    d = {"status": st, "cost": cost, "expanded": ex, "reopened": reo,
                         "sec": round(sec, 3), "rss_gb": round(rss_gb(), 2)}
                    for k in ("pre_sec", "unreach_nodes"):
                        if k in stat:
                            d[k] = stat[k]
                    if var.startswith("cea"):
                        d.update(h_calls=stat["calls"], h_sec=round(stat["sec"], 3))
                    if st == "ok":
                        J, ok, _ = verifier_J(sc, g, pipe, chain, std)
                        d.update(J_verifier=J, verifier_success=ok)
                    row[var] = d
                    del h
                    gc.collect()
            rows.append(row)
            with open(part, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            msg = " | ".join(f"{v} {row[v]['status'][:6]} {row[v]['expanded']}" for v in variants if v in row)
            print(f"  {name}/{pipe.id} {pipe.nominal_size}: 컴파일 {rb.status} {rb.expanded} | {msg}", flush=True)
        del sc
    summ = summarize(rows, variants)
    out = {"support_standard": std_info, "ckpt": a.ckpt, "variants": variants,
           "seeds": [seeds[0], seeds[-1]], "py_budget": a.py_budget, "summary": summ, "rows": rows}
    print("\n" + json.dumps(summ, ensure_ascii=False, indent=1))
    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return 0


def summarize(rows, variants) -> dict:
    """변형마다 base 와 짝지어 (둘 다 ok) R·품질·벽시계. C* = 컴파일 A* 비용."""
    out = {}
    for v in variants:
        ok = [r for r in rows if r.get(v, {}).get("status") == "ok"]
        s = {"ok": len(ok),
             "only_ok_vs_base": sum(r.get(v, {}).get("status") == "ok" and r.get("base", {}).get("status") != "ok"
                                    for r in rows),
             "base_only_ok": sum(r.get(v, {}).get("status") != "ok" and r.get("base", {}).get("status") == "ok"
                                 for r in rows),
             "memory_exhausted": sum(r.get(v, {}).get("status") == "memory_exhausted" for r in rows)}
        cs = [r for r in ok if r["compiled"]["status"] == "ok"]
        if cs:
            vc = [r[v]["cost"] / r["compiled"]["cost"] for r in cs]
            s.update(cost_vs_Cstar_max=round(max(vc), 5),
                     sum_cost_vs_Cstar=round(sum(r[v]["cost"] for r in cs) / sum(r["compiled"]["cost"] for r in cs), 5),
                     n_over_5pct_vs_Cstar=sum(x > 1.05 for x in vc), n_over_1pct_vs_Cstar=sum(x > 1.01 for x in vc))
        if v != "base":
            pair = [r for r in ok if r.get("base", {}).get("status") == "ok"]
            if pair:
                R = [r["base"]["expanded"] / r[v]["expanded"] for r in pair]
                jp = [r for r in pair if r["base"].get("J_verifier")]
                jr = [r[v]["J_verifier"] / r["base"]["J_verifier"] for r in jp]
                wall_b = sum(r["base"]["sec"] for r in pair)
                wall_v = sum(r[v]["sec"] + r[v].get("pre_sec", 0.0) for r in pair)
                s.update(n_paired=len(pair), R_median=round(statistics.median(R), 3),
                         R_p10=round(float(np.percentile(R, 10)), 3), R_p90=round(float(np.percentile(R, 90)), 3),
                         sum_J_ratio=round(sum(r[v]["J_verifier"] for r in jp) / sum(r["base"]["J_verifier"] for r in jp), 5),
                         J_ratio_max=round(max(jr), 5), n_J_over_5pct=sum(x > 1.05 for x in jr),
                         n_J_over_1pct=sum(x > 1.01 for x in jr),
                         wall_base=round(wall_b, 1), wall_var=round(wall_v, 1),
                         pre_sec=round(sum(r[v].get("pre_sec", 0.0) for r in pair), 1),
                         h_sec=round(sum(r[v].get("h_sec", 0.0) for r in pair), 1))
        out[v] = s
    return out


if __name__ == "__main__":
    raise SystemExit(main())
