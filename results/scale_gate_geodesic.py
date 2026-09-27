#!/usr/bin/env python3
"""§42 ③ 관문 — 거리장이 Dijkstra 와 일치 · octile 기본 경로 불변 · geodesic A* 가 검증 격자 1,096 탐색 전부 octile 과 같은 최적 J (_ab/_cache 에서 실행)."""
import sys, io, contextlib, time, json, heapq, math, numpy as np
sys.path.insert(0, "/home/user/pipe-routing-GNARL/results")
from run_step1_ab import load_cells, NB
ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
exec(compile("import os, sys, time", "<p>", "exec"), ns)
for tag, src in load_cells(NB):
    if tag in ("3.", "4."):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.": ns["cfg"].pretrain_h=False; ns["cfg"].bc_batches=0
cfg = ns["cfg"]; pool = ns["TRAIN_POOL"] + ns["EVAL_POOL"]; D26 = ns["DIRS26"]
def dijkstra(free, goal):
    d = np.full(free.shape, np.inf); d[goal] = 0; h = [(0.0, goal)]
    while h:
        c, u = heapq.heappop(h)
        if c > d[u]: continue
        for dx, dy, dz in D26:
            v = (u[0]+dx, u[1]+dy, u[2]+dz)
            if 0 <= v[0] < free.shape[0] and 0 <= v[1] < free.shape[1] and 0 <= v[2] < free.shape[2] and free[v]:
                nc = c + math.sqrt(dx*dx+dy*dy+dz*dz)
                if nc < d[v] - 1e-12: d[v] = nc; heapq.heappush(h, (nc, v))
    return d
out = {"field_vs_dijkstra": []}
for sc in pool[:5]:
    t0=time.time(); f, rounds = ns["geodesic_field"](sc["free"], tuple(sc["goal"])); tf=time.time()-t0
    ref = dijkstra(sc["free"], tuple(sc["goal"]))
    fin = np.isfinite(ref)
    out["field_vs_dijkstra"].append(dict(maxdiff=float(np.max(np.abs(f[fin]-ref[fin]))), inf_match=bool(np.array_equal(np.isfinite(f), fin)), rounds=rounds, sec=round(tf,3)))
# octile default unchanged
oc = []
for sc in pool[:60]:
    for d, sol in sc["astar"].items():
        r = ns["astar_gnarl"](sc["pad"], sc["start"], sc["goal"], d)
        oc.append(abs(r["J"]-sol["J"]) < 1e-9 and r["expanded"] == sol["expanded"])
out["octile_default_identical_60scen"] = all(oc)
cfg.astar_heuristic = "geodesic"
rows = []; t0 = time.time()
for sc in pool:
    for d, sol in sc["astar"].items():
        r = ns["astar_gnarl"](sc["pad"], sc["start"], sc["goal"], d)
        rows.append((sol["J"], r["J"] if r else None, sol["expanded"], r["expanded"] if r else None))
J_ok = all(b is not None and abs(a-b) < 1e-9 for a, b, _, _ in rows)
ratio = np.array([c/max(1,e) for _, _, e, c in rows if c])
out.update(geo_all_J_equal=J_ok, n_searches=len(rows), sec=round(time.time()-t0,1),
           exp_octile_mean=float(np.mean([r[2] for r in rows])), exp_geo_mean=float(np.mean([r[3] for r in rows])),
           exp_ratio_median=float(np.median(ratio)), exp_ratio_p90=float(np.percentile(ratio,90)))
json.dump(out, open("gate3.json","w"), indent=1); print(json.dumps(out, indent=1))
