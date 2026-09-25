#!/usr/bin/env python3
"""§31-7 판정 — 분할별 짝지은 Δ(인코딩 − 80차원 기준) 의 평균이 2SE 이상 양수인가. 선등록 기준 그대로.
사용: python3 judge_h2_precheck.py results/step1_h2_precheck.json [--write]"""
import json, sys
from statistics import mean, stdev
p = sys.argv[1]; d = json.load(open(p)); R = d["results"]
base = {r["split"]: r for r in R["base"]}
gate = base[0]["heldout"]
out = dict(gate_split0=gate, gate_ok=abs(gate - 0.721) <= 0.01, arms={})
print(f"건전성 관문: 기준 분할 0 보류 top-1 {gate:.4f} (§28-5 0.721 ± 0.01) → {'통과' if out['gate_ok'] else '실패'}")
print(f"기준 보류 top-1 {mean(r['heldout'] for r in R['base']):.4f} ± "
      f"{stdev([r['heldout'] for r in R['base']]) / len(R['base']) ** .5:.4f} (분할 {len(R['base'])}개)")
for arm, nm in (("a", "H2-a 원거리 광선"), ("b", "H2-b 블록 점유율"), ("c", "H2-c 지오데식")):
    rs = [r for r in R.get(arm, []) if r["split"] in base]
    if len(rs) < 2:
        continue
    dl = [r["heldout"] - base[r["split"]]["heldout"] for r in rs]
    m, se = mean(dl), stdev(dl) / len(dl) ** .5
    ho = [r["heldout"] for r in rs]; gap = mean(r["train"] - r["heldout"] for r in rs)
    passed = m > 0 and m >= 2 * se
    out["arms"][arm] = dict(name=nm, delta=dl, mean=m, se=se, heldout=ho, heldout_mean=mean(ho),
                            vs_0721=mean(ho) - 0.721, train_gap=gap, passed=passed)
    print(f"{nm:<16} 보류 {mean(ho):.4f} (0.721 대비 {mean(ho)-0.721:+.4f}) · Δ {m:+.4f} ± {se:.4f} "
          f"({'2SE 통과' if passed else '2SE 안쪽'}) · 학습-보류 격차 {gap:.3f} · 분할별 {['%+.3f' % x for x in dl]}")
win = [a for a, v in out["arms"].items() if v["passed"]]
if not out["gate_ok"]:
    call = "측정 중단 — 건전성 관문 실패"
elif win:
    best = max(win, key=lambda a: out["arms"][a]["mean"])
    call = f"R-OBS 1회 해제 후보: {out['arms'][best]['name']}" + (f" (그 밖의 통과: {', '.join(out['arms'][a]['name'] for a in win if a != best)})" if len(win) > 1 else "")
else:
    call = "관측 가설 소진 — §7 결정은 (i)/(iii) 중에서"
out["call"] = call
print("\n§31-7 판정:", call)
if "--write" in sys.argv:
    d["judge"] = out; json.dump(d, open(p, "w"), ensure_ascii=False, indent=1)
