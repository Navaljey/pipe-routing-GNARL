#!/usr/bin/env python3
"""run_step1_ab.py 의 런 결과를 §11 프로토콜대로 모아 본다 (시드 평균 + 산포).

사용: python3 summarize_step1_ab.py [--root ../_ab] [--json out.json]
"""
import argparse, glob, json, os
from statistics import mean, stdev

def se(v):
    return (stdev(v) / len(v) ** 0.5) if len(v) > 1 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_ab"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    runs = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "run_result.json"))):
        r = json.load(open(f))
        r["dir"] = os.path.basename(os.path.dirname(f))
        runs.append(r)

    by_arm = {}
    for r in runs:
        by_arm.setdefault((r["arm"], r["steps"]), []).append(r)

    out = {}
    for (arm, steps), rs in sorted(by_arm.items()):
        rs.sort(key=lambda r: r["seed"])
        pts = sorted({h["timesteps"] for r in rs for h in r["history"]})
        print(f"\n===== arm {arm} | {steps:,} steps | seeds {[r['seed'] for r in rs]} "
              f"| override {rs[0]['override']}")
        print(f"{'step':>9} | {'success mean±SE':>18} | {'tau mean':>9} | {'h MAE kg':>9} | "
              f"{'entropy':>7} | {'maxp':>5} | per-seed")
        rows = []
        for p in pts:
            vals = [h for r in rs for h in r["history"] if h["timesteps"] == p]
            if not vals:
                continue
            s = [v["success_rate"] for v in vals]
            tau = [v["tau"] for v in vals if "tau" in v]
            hm = [v["h_mae_kg"] for v in vals if "h_mae_kg" in v]
            en = [v["entropy"] for v in vals if v.get("entropy") == v.get("entropy")]
            mp = [v["max_prob"] for v in vals if v.get("max_prob") == v.get("max_prob")]
            row = dict(timesteps=p, n=len(s), success=mean(s), success_se=se(s),
                       tau=mean(tau) if tau else None, h_mae_kg=mean(hm) if hm else None,
                       entropy=mean(en) if en else None, max_prob=mean(mp) if mp else None,
                       per_seed=s)
            rows.append(row)
            print(f"{p:>9,} | {mean(s):>9.3f} ± {se(s):<6.3f} | "
                  f"{(mean(tau) if tau else float('nan')):>9.4f} | "
                  f"{(mean(hm) if hm else float('nan')):>9.2f} | "
                  f"{(mean(en) if en else float('nan')):>7.2f} | "
                  f"{(mean(mp) if mp else float('nan')):>5.2f} | "
                  + " ".join(f"{x:.3f}" for x in s))
        fin = [r["final"] for r in rs]
        print(f"{'final':>9} | {mean([f['success_rate'] for f in fin]):>9.3f} ± "
              f"{se([f['success_rate'] for f in fin]):<6.3f} | "
              f"length_ratio {mean([f['length_ratio'] for f in fin]):.2f} | "
              f"deadlock {mean([f['deadlock_rate'] for f in fin]):.3f} | "
              f"timeout {mean([f['timeout_rate'] for f in fin]):.3f} | "
              f"minutes {mean([r['minutes'] for r in rs]):.1f}")
        out[f"{arm}_{steps}"] = dict(arm=arm, steps=steps, seeds=[r["seed"] for r in rs],
                                     override=rs[0]["override"], curve=rows,
                                     final=dict(success=mean([f["success_rate"] for f in fin]),
                                                success_se=se([f["success_rate"] for f in fin]),
                                                length_ratio=mean([f["length_ratio"] for f in fin]),
                                                deadlock=mean([f["deadlock_rate"] for f in fin]),
                                                timeout=mean([f["timeout_rate"] for f in fin]),
                                                h_mae_kg=mean([f["h_mae_kg"] for f in fin]),
                                                tau=mean([f["tau"] for f in fin if "tau" in f])
                                                if any("tau" in f for f in fin) else None))
    # 짝지은 비교 (같은 시드끼리) — §11
    keys = sorted(by_arm)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (a1, st1), (a2, st2) = keys[i], keys[j]
            if st1 != st2:
                continue
            m1 = {r["seed"]: r["final"]["success_rate"] for r in by_arm[keys[i]]}
            m2 = {r["seed"]: r["final"]["success_rate"] for r in by_arm[keys[j]]}
            common = sorted(set(m1) & set(m2))
            if len(common) < 2:
                continue
            d = [m2[s] - m1[s] for s in common]
            print(f"\n짝지은 비교 {a2} - {a1} ({st1:,} steps, seeds {common}): "
                  f"Δ = {mean(d):+.3f} ± {se(d):.3f} (SE) | " + " ".join(f"{x:+.3f}" for x in d))
            out.setdefault("paired", []).append(dict(a=a1, b=a2, steps=st1, seeds=common,
                                                     delta=mean(d), se=se(d), per_seed=d))
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1, default=float)
        print(f"\n→ {a.json}")

if __name__ == "__main__":
    main()
