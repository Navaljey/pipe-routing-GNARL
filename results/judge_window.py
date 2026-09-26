#!/usr/bin/env python3
"""§34-0 판정 — 시드마다 이력의 마지막 N(기본 4)개 평가 평균으로 짝지은 비교.

사용: python3 judge_window.py --ref "_ab/L_300k_bc_s*" --arm bc1k="_ab/L_300k_bc1k_s*" --arm bc6k=... [--out x.json]
     기준(ref) 대비 각 arm 의 짝지은 Δ ± SE 와 2SE 판정을 낸다. 종료 평가 한 점의 값도 함께 적는다 (판정에는 안 쓴다).
"""
import argparse, glob, json, re
import numpy as np

KEYS = ["j_ratio", "success_rate", "elbow_ratio", "deadlock_rate", "astar_overlap", "approach_steps",
        "rank_top1", "rank_spearman"]


def load(pattern):
    out = {}
    for d in sorted(glob.glob(pattern)):
        m = re.search(r"_s(\d+)$", d)
        f = f"{d}/run_result.json"
        try:
            out[int(m.group(1))] = json.load(open(f))["history"]
        except Exception:
            pass
    return out


def summ(hist, n):
    last = hist[-n:]
    w = {k: float(np.nanmean([h.get(k, np.nan) for h in last])) for k in KEYS}
    f = {k: float(hist[-1].get(k, np.nan)) for k in KEYS}
    return w, f, [h["timesteps"] for h in last]


def paired(a, b):
    d = np.array(b) - np.array(a)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se, d.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-name", default="ref")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = load(a.ref)
    res = dict(n_last=a.n, ref=a.ref_name, arms={})
    rw = {s: summ(h, a.n) for s, h in ref.items()}
    print(f"기준 {a.ref_name}: 시드 {sorted(ref)} · 창 {rw[min(rw)][2]}")
    for spec in a.arm:
        name, pat = spec.split("=", 1)
        arm = load(pat)
        seeds = sorted(set(arm) & set(ref))
        aw = {s: summ(arm[s], a.n) for s in seeds}
        row = dict(seeds=seeds, window={}, final={})
        print(f"\n== {name} (시드 {seeds}, 창 {aw[seeds[0]][2] if seeds else '-'})")
        print(f"{'지표':16s} {a.ref_name:>8s} {name:>8s}   Δ(창) ± SE    2SE    | Δ(종료 한 점)")
        for k in KEYS:
            ra = [rw[s][0][k] for s in seeds]; aa = [aw[s][0][k] for s in seeds]
            if np.all(np.isnan(ra)) or np.all(np.isnan(aa)):
                continue
            m, se, d = paired(ra, aa)
            fm, fse, _ = paired([rw[s][1][k] for s in seeds], [aw[s][1][k] for s in seeds])
            tag = "감소" if m < -2 * se else ("증가" if m > 2 * se else "노이즈")
            print(f"{k:16s} {np.mean(ra):8.3f} {np.mean(aa):8.3f}  {m:+.3f} ± {se:.3f}  {tag:4s}  | {fm:+.3f} ± {fse:.3f}")
            row["window"][k] = dict(ref=float(np.mean(ra)), arm=float(np.mean(aa)), delta=m, se=se, per_seed=d, call=tag,
                                    arm_se=float(np.std(aa, ddof=1) / np.sqrt(len(aa))))
            row["final"][k] = dict(delta=fm, se=fse)
        res["arms"][name] = row
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
