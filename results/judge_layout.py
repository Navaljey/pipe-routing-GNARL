#!/usr/bin/env python3
"""§42-3-1 판정 — 먼지 ÷ 큰 상자 A* 전개 수의 기하평균 비 + 계층 붓스트랩 95% 구간 (격자 → 쌍, 10,000회, 시드 0).
사용: python3 results/judge_layout.py results/step1_astar_layout_dust.json results/step1_astar_layout_bigbox.json --out results/step1_astar_layout.json"""
import argparse, json
import numpy as np


def groups(rows, dist):
    g = {}
    for r in rows:
        if r["target"] == dist:
            g.setdefault(r["grid"], []).append(np.log10(r["expanded"]))
    return [np.array(v) for _, v in sorted(g.items())]


def boot_mean(gs, rng):
    idx = rng.integers(len(gs), size=len(gs))
    return np.mean(np.concatenate([gs[i][rng.integers(len(gs[i]), size=len(gs[i]))] for i in idx]))


def summarize(rows, dist, lim):
    rs = [r for r in rows if r["target"] == dist]
    e = np.array([r["expanded"] for r in rs], float)
    return dict(n=len(rs), n_grids=len({r["grid"] for r in rs}), ok=int(sum(r["ok"] for r in rs)),
                within_600k=int((e <= 600_000).sum()), geo_mean=float(10 ** np.log10(e).mean()),
                median=float(np.median(e)), min=float(e.min()), max=float(e.max()),
                sec_median=float(np.median([r["sec"] for r in rs])), sec_max=float(max(r["sec"] for r in rs)),
                rss_growth_max_mb=int(max(r["rss_growth_mb"] for r in rs)),
                exp_per_cell_median=(float(np.median([r["expanded"] / r["n_cells"] for r in rs if r["ok"]]))
                                     if any(r["ok"] for r in rs) else None),
                manhattan_mean=float(np.mean([r["manhattan"] for r in rs])),
                failed_at_limit=int(sum((not r["ok"]) for r in rs)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dust"); ap.add_argument("bigbox"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args()
    D, B = json.load(open(a.dust)), json.load(open(a.bigbox))
    out = dict(dust_grids=D["grids"], bigbox_grids=B["grids"], by_dist={})
    for dist in (270, 500):
        gd, gb = groups(D["rows"], dist), groups(B["rows"], dist)
        if not gd or not gb:
            continue
        rng = np.random.default_rng(0)
        bs = np.array([boot_mean(gd, rng) - boot_mean(gb, rng) for _ in range(a.n_boot)])
        point = np.mean(np.concatenate(gd)) - np.mean(np.concatenate(gb))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        R = dict(ratio=float(10 ** point), ci95=[float(10 ** lo), float(10 ** hi)],
                 dust=summarize(D["rows"], dist, D["max_expand"]), bigbox=summarize(B["rows"], dist, B["max_expand"]),
                 dust_ratio_is_lower_bound=bool(any((not r["ok"]) for r in D["rows"] if r["target"] == dist)))
        if dist == 500:
            R["verdict"] = "배치 의존 확인" if 10 ** lo >= 10 else "확인 안 됨"
        out["by_dist"][str(dist)] = R
        print(f"~{dist}: 비 {R['ratio']:.1f}× (95% {R['ci95'][0]:.1f}~{R['ci95'][1]:.1f}) {R.get('verdict','')}")
        for k in ("dust", "bigbox"):
            s = R[k]; print(f"   {k:6s} n={s['n']} ok={s['ok']} ≤60만={s['within_600k']} 기하평균 {s['geo_mean']:,.0f} "
                             f"중앙 {s['median']:,.0f} [{s['min']:,.0f}~{s['max']:,.0f}] 초 중앙 {s['sec_median']} 최대 {s['sec_max']}")
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("→", a.out)


if __name__ == "__main__":
    main()
