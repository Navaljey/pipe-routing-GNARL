"""§33 관문 판정 — out/cf_v1.json(.partial.jsonl) → 표.

    python3 v3/report_cf.py out/cf_v1.json.partial.jsonl
"""

from __future__ import annotations

import json
import statistics as st
import sys

import numpy as np


def main(path: str) -> int:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    keys = list(rows[0]["v"].keys())
    pair = [r for r in rows if r["v"]["base"]["status"] == "ok"]
    print(f"배관 {len(rows)} · base 성공 {len(pair)}")
    W = lambda v: v["sec"] + v["pre_sec"]                                   # noqa: E731
    print(f"{'변형':<20}{'ok':>4}{'잃음':>5}{'Σ벽시계':>9}{'비(base÷)':>10}{'비(rg÷)':>9}"
          f"{'R′중앙':>8}{'ΣJ비':>9}{'+1%':>5}{'+5%':>5}{'표초':>7}{'탐색만비(rg÷)':>14}")
    rgW = {id(r): W(r["v"]["rg1.0"]) for r in pair}
    for k in keys:
        p = [r for r in pair if r["v"][k]["status"] == "ok"]
        lost = len(pair) - len(p)
        both = [r for r in p if r["v"]["rg1.0"]["status"] == "ok"]
        sw = sum(W(r["v"][k]) for r in p)
        wb = st.median(r["v"]["base"]["sec"] / W(r["v"][k]) for r in p)
        wr = st.median(rgW[id(r)] / W(r["v"][k]) for r in both)
        rr = st.median(r["v"]["rg1.0"]["expanded"] / max(r["v"][k]["expanded"], 1) for r in both)
        jr = [r["v"][k]["J_verifier"] / r["v"]["base"]["J_verifier"] for r in p]
        sj = sum(r["v"][k]["J_verifier"] for r in p) / sum(r["v"]["base"]["J_verifier"] for r in p)
        tab = sum(r["v"][k]["pre_sec"] - r["v"]["rg1.0"]["pre_sec"] for r in p)
        so = st.median(r["v"]["rg1.0"]["sec"] / max(r["v"][k]["sec"], 1e-6) for r in both)
        print(f"{k:<20}{len(p):>4}{lost:>5}{sw:>9.1f}{wb:>10.2f}{wr:>9.2f}{rr:>8.2f}{sj:>9.4f}"
              f"{sum(j > 1.01 for j in jr):>5}{sum(j > 1.05 for j in jr):>5}{tab:>7.1f}{so:>14.2f}")
    # 관문 C-1 / C-2 (§33.2) — 짝 = base 성공. C-1 은 rg1.0 과 cf 가 둘 다 성공한 배관
    print("\n관문 (§33.2)")
    rgsum = None
    for k in keys:
        if not k.startswith("cf_"):
            continue
        p = [r for r in pair if r["v"][k]["status"] == "ok" and r["v"]["rg1.0"]["status"] == "ok"]
        med = st.median(W(r["v"]["rg1.0"]) / W(r["v"][k]) for r in p)
        s_cf = sum(W(r["v"][k]) for r in p)
        s_rg = sum(W(r["v"]["rg1.0"]) for r in p)
        q = [r for r in pair if r["v"][k]["status"] == "ok"]
        sj = sum(r["v"][k]["J_verifier"] for r in q) / sum(r["v"]["base"]["J_verifier"] for r in q)
        o5 = sum(r["v"][k]["J_verifier"] / r["v"]["base"]["J_verifier"] > 1.05 for r in q)
        lost = len(pair) - len(q)
        c1 = med > 1.0 and s_cf < s_rg
        c2 = sj <= 1.01 and o5 == 0 and lost == 0
        print(f"  {k}: C-1 중앙 {med:.3f} · Σ {s_cf:.1f} vs rg {s_rg:.1f} → {'✅' if c1 else '❌'} | "
              f"C-2 ΣJ {sj:.4f} · +5% {o5} · 잃음 {lost} → {'✅' if c2 else '❌'} | 통과 {'✅' if c1 and c2 else '❌'}")
    # 관문 C-3 (§33.4) — w1.1 변형 대 rg1.1
    c3 = [k for k in keys if k.startswith("cf_") and k.endswith("_w1.1")]
    if c3:
        print("\n관문 C-3 (§33.4, rg1.1 대비)")
        for k in c3:
            p = [r for r in pair if r["v"][k]["status"] == "ok" and r["v"]["rg1.1"]["status"] == "ok"]
            med = st.median(W(r["v"]["rg1.1"]) / W(r["v"][k]) for r in p)
            s_cf = sum(W(r["v"][k]) for r in p)
            s_rg = sum(W(r["v"]["rg1.1"]) for r in p)
            q = [r for r in pair if r["v"][k]["status"] == "ok"]
            sj = sum(r["v"][k]["J_verifier"] for r in q) / sum(r["v"]["base"]["J_verifier"] for r in q)
            o5 = sum(r["v"][k]["J_verifier"] / r["v"]["base"]["J_verifier"] > 1.05 for r in q)
            lost = len(pair) - len(q)
            ok1 = med > 1.0 and s_cf < s_rg
            ok2 = sj <= 1.01 and o5 == 0 and lost == 0
            wins = sum(W(r["v"]["rg1.1"]) > W(r["v"][k]) for r in p)
            print(f"  {k}: 중앙 {med:.3f} · Σ {s_cf:.1f} vs rg1.1 {s_rg:.1f} · 이긴 배관 {wins}/{len(p)} → {'✅' if ok1 else '❌'} | "
                  f"C-2 ΣJ {sj:.4f} · +5% {o5} · 잃음 {lost} → {'✅' if ok2 else '❌'} | 통과 {'✅' if ok1 and ok2 else '❌'}")
    # 구간별 (기준 시간)
    print("\n기준 시간 구간별 Σ벽시계 (짝 = base·rg1.0·cf 전부 성공)")
    bins = [(0, 0.1), (0.1, 1), (1, 10), (10, 1e9)]
    cfk = [k for k in keys if k.startswith("cf_")]
    for lo, hi in bins:
        b = [r for r in pair if lo <= r["v"]["base"]["sec"] < hi
             and all(r["v"][k]["status"] == "ok" for k in ["rg1.0", "rg1.1"] + cfk)]
        if not b:
            continue
        s = " · ".join(f"{k} {sum(W(r['v'][k]) for r in b):.1f}" for k in ["base", "rg1.0", "rg1.1"] + cfk)
        tb = " · ".join(f"{k} 표 {sum(r['v'][k]['pre_sec'] - r['v']['rg1.0']['pre_sec'] for r in b):.1f}" for k in cfk)
        print(f"  [{lo}, {hi}) n={len(b)}: {s}\n      {tb}")
    # 표 비용
    for k in cfk:
        t = [r["v"][k].get("table") for r in rows if r["v"][k].get("table")]
        print(f"\n{k} 표: 예측 노드 중앙 {st.median(x['n_pred'] for x in t):,.0f} / 전체 {st.median(x['n_nodes'] for x in t):,.0f}"
              f" · 특징 중앙 {st.median(x['feat_sec'] for x in t):.3f}s · MLP 중앙 {st.median(x['mlp_sec'] for x in t):.3f}s")
    # 성공 수 (base 실패 배관 포함)
    print("\n성공 수 (100 중): " + " · ".join(f"{k} {sum(r['v'][k]['status'] == 'ok' for r in rows)}" for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
