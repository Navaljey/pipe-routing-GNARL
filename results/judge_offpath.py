#!/usr/bin/env python3
"""§28 (b) 판정 — H3fix(대조) vs 경로 밖 라벨(b), 같은 시드 짝지은 Δ ± SE, 2SE 기준 (사전 등록 그대로).

  채택   : J_ratio(성공분) 2SE 이상 감소 AND success 2SE 이상 하락 없음
  기각   : J_ratio 변화 2SE 안쪽
  §1 원칙 위반 신호 (§28-3): astar_overlap 2SE 이상 상승 AND J_ratio 변화 2SE 안쪽 → 판정과 별도 보고, 채택 안 함
두 arm 모두 학습 종료 평가(history 마지막, 같은 코드 · 같은 평가 풀)를 쓴다.
사용: python3 judge_offpath.py "_ab/L_300k_h3fix_s*" "_ab/L_300k_offpath_s*" [--out results/step1_offpath.json]
"""
import glob, json, sys
from statistics import mean, stdev


def last(pattern):
    out = {}
    for d in sorted(glob.glob(pattern)):
        out[int(d.rsplit("_s", 1)[1])] = json.load(open(f"{d}/step1_history.json"))["history"][-1]
    return out


A, B = last(sys.argv[1]), last(sys.argv[2])
seeds = sorted(set(A) & set(B))
se = lambda v: stdev(v) / len(v) ** .5 if len(v) > 1 else float("nan")
KEYS = [("j_ratio", "J_ratio 성공분 (주)", -1), ("elbow_ratio", "elbow_ratio", -1), ("success_rate", "success", +1),
        ("steps_approach", "진입 구간 스텝", -1), ("astar_overlap", "A* 경로 겹침률 (§28-3)", 0),
        ("j_ratio_all", "J_ratio 전체 (계기판)", -1), ("deadlock_rate", "deadlock", -1),
        ("rank_top1", "top-1", +1), ("rank_spearman", "rho", +1), ("hard_gap", "3분위 어려움 격차", -1)]
out = dict(seeds=seeds, rows={})
print(f"시드 {seeds} · 대조 @ {[A[s]['timesteps'] for s in seeds]} · (b) @ {[B[s]['timesteps'] for s in seeds]}")
print(f"{'지표':<24}{'H3fix':>9}{'(b)':>9}{'Δ':>9}{'SE':>8}   2SE")
for k, nm, good in KEYS:
    a = [A[s].get(k, float("nan")) for s in seeds]; b = [B[s].get(k, float("nan")) for s in seeds]
    d = [y - x for x, y in zip(a, b) if x == x and y == y]
    if len(d) < 2:
        continue
    m, e = mean(d), se(d)
    sig = m != 0 and abs(m) >= 2 * e
    vd = ("상승" if m > 0 else "하락") if (sig and good == 0) else (("개선" if m * good > 0 else "악화") if sig else "노이즈")
    out["rows"][k] = dict(ctrl=a, arm=b, delta=d, mean=m, se=e, verdict=vd)
    print(f"{nm:<24}{mean(a):9.3f}{mean(b):9.3f}{m:+9.3f}{e:8.3f}   {vd}  {['%+.3f' % x for x in d]}")
R = out["rows"]
jr, sc, ov = R["j_ratio"]["verdict"], R["success_rate"]["verdict"], R["astar_overlap"]["verdict"]
signal = (ov == "상승" and jr == "노이즈")
if signal:
    call = "기각 — §1 원칙 위반 신호 (겹침만 오르고 J_ratio 정체)"
elif sc == "악화":
    call = "해로움"
elif jr == "개선":
    call = "채택"
else:
    call = "기각"
out.update(call=call, principle_signal=signal)
print(f"\n§28-2 판정: {call}")
print(f"§28-3 감시: 겹침률 {ov} · J_ratio {jr} → " + ("⚠ §1 원칙 위반 신호" if signal else "신호 없음"))
if "--out" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), ensure_ascii=False, indent=1)
