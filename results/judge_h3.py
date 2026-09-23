#!/usr/bin/env python3
"""§25-2 H3 판정 — h1k 300K(대조) vs H3 300K, 같은 시드 짝지은 Δ ± SE, 2SE 기준.

대조군은 regrade.py 로 **현재 코드**에서 재평가한 값(진입 구간 지표 포함)을 쓰고,
H3 는 학습 종료 평가(history 마지막)를 쓴다 — 둘 다 같은 evaluate_routing, 같은 평가 풀.
사용: python3 judge_h3.py <대조 regrade.json> <H3 런 디렉터리 glob> [--out out.json]
"""
import glob, json, sys
from statistics import mean, stdev

ctrl = json.load(open(sys.argv[1]))["per_run"]
h3 = {}
for d in sorted(glob.glob(sys.argv[2])):
    h = json.load(open(f"{d}/step1_history.json"))["history"]
    h3[int(d.rsplit("_s", 1)[1])] = h[-1]
c = {int(k.rsplit("_s", 1)[1]): v for k, v in ctrl.items()}
seeds = sorted(set(c) & set(h3))
se = lambda v: stdev(v) / len(v) ** .5 if len(v) > 1 else float("nan")
KEYS = [("steps_approach", "진입 구간 스텝 (방향)", -1), ("j_ratio", "J_ratio 성공분 (주)", -1),
        ("elbow_ratio", "elbow_ratio 성공분", -1), ("success_rate", "success", +1),
        ("reentries", "재진입 (성공)", -1), ("reentries_fail", "재진입 (실패)", -1),
        ("j_ratio_all", "J_ratio 전체 (계기판)", -1), ("deadlock_rate", "deadlock", -1),
        ("rank_top1", "top-1", +1), ("rank_spearman", "rho", +1)]
out = {"seeds": seeds, "rows": {}}
print(f"시드 {seeds} · 대조 @ {[c[s]['timesteps'] for s in seeds]} · H3 @ {[h3[s]['timesteps'] for s in seeds]}")
print(f"{'지표':<22}{'대조':>9}{'H3':>9}{'Δ':>9}{'SE':>8}   판정(2SE)")
for k, nm, good in KEYS:
    if not all(k in c[s] and k in h3[s] for s in seeds):
        continue
    a = [c[s][k] for s in seeds]; b = [h3[s][k] for s in seeds]
    dd = [y - x for x, y in zip(a, b)]
    m, e = mean(dd), se(dd)
    sig = abs(m) >= 2 * e
    verdict = ("개선" if m * good > 0 else "악화") if sig else "노이즈"
    out["rows"][k] = dict(ctrl=a, h3=b, delta=dd, mean=m, se=e, verdict=verdict)
    print(f"{nm:<22}{mean(a):9.3f}{mean(b):9.3f}{m:+9.3f}{e:8.3f}   {verdict}  {['%+.3f' % x for x in dd]}")
print(f"A* 진입 구간 스텝: {mean(c[s]['steps_approach_base'] for s in seeds):.1f}")
R = out["rows"]
jr, ap, sc = R["j_ratio"]["verdict"], R["steps_approach"]["verdict"], R["success_rate"]["verdict"]
if sc == "악화": call = "해로움"
elif jr == "개선": call = "채택"
elif ap == "개선": call = "부분"
else: call = "기각"
out["call"] = call
print(f"\n§25-2 판정: {call}" + ("" if call == "채택" else "  → §25-3: S1-L2 직행 (채널 재설계 금지)"))
if "--out" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), ensure_ascii=False, indent=1)
