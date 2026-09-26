#!/usr/bin/env python3
"""slope_feasibility.py 원자료 요약 (§36). 사용: python3 summarize_slope.py step1_slope_feasibility.json"""
import json, sys
import numpy as np

d = json.load(open(sys.argv[1]))
rules = d["rules"]
G = [r for r in rules if r != "P"]
out = {"wall_sec": d["wall_sec"]}


def ok(res, r):
    return res.get(r, {}).get("status") == "ok"


for pool in ("train", "eval", "all"):
    rows = [r for r in d["rows"] if pool == "all" or r["pool"] == pool]
    n = len(rows)
    if not n:
        continue
    o = [r["orig"] for r in rows]
    dz = np.array([r["dz"] for r in rows])
    S = {"n": n, "dz_ge0": float((dz >= 0).mean()), "dz_eq0": float((dz == 0).mean())}
    S["P_cache_maxdiff"] = float(max(abs(x["P"]["J"] - x["P_cache_J"]) for x in o))
    ps = [x["P_stats"] for x in o]
    m = dz >= 0
    S["P_noclimb_given_dzge0"] = float(np.mean([p["climbs"] == 0 for p, k in zip(ps, m) if k]))
    S["P_maxrun_pct"] = {q: float(np.percentile([p["maxrun"] for p in ps], q)) for q in (50, 90, 100)}
    S["P_maxrun_gt"] = {L: float(np.mean([p["maxrun"] > L for p in ps])) for L in (25, 50, 100)}
    S["P_H_mean"] = float(np.mean([p["H"] for p in ps]))
    S["P_expanded_mean"] = float(np.mean([x["P"]["expanded"] for x in o]))
    per = {}
    for r in G:
        f = np.array([ok(x, r) for x in o])
        lim = sum(x.get(r, {}).get("status") == "limit" for x in o)
        jr = [x[r]["J"] / x["P"]["J"] for x in o if ok(x, r)]
        div = [x[r]["J"] > x["P"]["J"] + 1e-6 for x in o if ok(x, r)]
        exp = [x[r]["expanded"] / x["P"]["expanded"] for x in o if ok(x, r)]
        # 교체: 원래 방향이 불가능한 시나리오 중 흐름을 뒤집으면 되는 비율
        sw = [ok(rr["swap"], r) for rr, fo in zip(rows, f) if not fo and isinstance(rr["swap"], dict)]
        either = [fo or ok(rr["swap"], r) for rr, fo in zip(rows, f)]
        per[r] = dict(feasible=float(f.mean()), feasible_given_dzge0=float(f[m].mean()) if m.any() else None,
                      limit=int(lim), J_ratio_mean=float(np.mean(jr)) if jr else None,
                      J_ratio_max=float(np.max(jr)) if jr else None,
                      diverge=float(np.mean(div)) if div else None,
                      expand_ratio_median=float(np.median(exp)) if exp else None,
                      swap_rescue=float(np.mean(sw)) if sw else None,
                      either=float(np.mean(either)))
    S["rules"] = per
    out[pool] = S

json.dump(out, open(sys.argv[1].replace(".json", "_summary.json"), "w"), indent=1)
for pool in [p for p in ("eval", "train", "all") if p in out]:
    S = out[pool]
    print(f"\n== {pool} n={S['n']}  dz>=0 {S['dz_ge0']:.3f} (dz=0 {S['dz_eq0']:.3f})  P-cache maxdiff {S['P_cache_maxdiff']:.1e}")
    print(f"   P: no-climb|dz>=0 {S['P_noclimb_given_dzge0']:.3f}  maxrun p50/p90/max {S['P_maxrun_pct']}  maxrun>25/50/100 {S['P_maxrun_gt']}  H {S['P_H_mean']:.1f}")
    print(f"   {'rule':8s} feas  feas|dz>=0  lim  Jratio(mean/max)  diverge  exp×  swap-rescue  either")
    for r, v in S["rules"].items():
        f = lambda x, k=3: "  -  " if x is None else f"{x:.{k}f}"
        print(f"   {r:8s} {v['feasible']:.3f}  {f(v['feasible_given_dzge0'])}      {v['limit']:3d}  "
              f"{f(v['J_ratio_mean'],4)}/{f(v['J_ratio_max'],3)}  {f(v['diverge'])}  {f(v['expand_ratio_median'],2)}  "
              f"{f(v['swap_rescue'])}  {v['either']:.3f}")
