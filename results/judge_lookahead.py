#!/usr/bin/env python3
"""§26-2 S1-L2 단계 1 판정 — 같은 체크포인트의 1-ply 대비 짝지은 Δ ± SE (시드 4개), 2SE 기준.

사용: python3 judge_lookahead.py results/_la/h3_s*.json [--out results/step1_s1l2_lookahead.json]
"""
import glob, json, sys
from statistics import mean, stdev

files = sorted(f for a in sys.argv[1:] if not a.startswith("--") and a.endswith(".json") for f in glob.glob(a))
if "--out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--out") + 1]
    files = [f for f in files if f != out_path]
else:
    out_path = None
R = [json.load(open(f)) for f in files]
seeds = [r["run"] for r in R]
se = lambda v: stdev(v) / len(v) ** .5 if len(v) > 1 else float("nan")
ARMS = ["2ply", "beam4", "macro2"]
KEYS = [("j_ratio", "J_ratio 전체 (주)", -1), ("jr_cruise", "J_ratio 순항", -1), ("jr_approach", "J_ratio 진입", -1),
        ("exc_total", "초과 kg 전체", -1), ("exc_cruise", "초과 kg 순항 (핵심)", -1), ("exc_approach", "초과 kg 진입", -1),
        ("elbow_ratio", "elbow_ratio", -1), ("steps_approach", "진입 구간 스텝", -1),
        ("success_rate", "success", +1), ("deadlock_rate", "deadlock", -1)]


def verdict(d, good):
    m, e = mean(d), se(d)
    if e != e:
        return m, e, "판정불가"
    if m == 0 or abs(m) < 2 * e:
        return m, e, "노이즈"
    return m, e, ("개선" if m * good > 0 else "악화")


base = {k: [r["methods"]["1ply"]["agg"][k] for r in R] for k, _, _ in KEYS}
print(f"시드 {seeds}")
print(f"건전성: 정책 vs h 기반 1-ply 불일치 "
      + ", ".join(f"{r['methods']['1ply']['agg']['policy_vs_h1ply_mismatch']}/"
                  f"{r['methods']['1ply']['agg']['policy_vs_h1ply_compared']}" for r in R))
print(f"\n1-ply (대조): " + " · ".join(f"{nm} {mean(base[k]):.3f}" for k, nm, _ in KEYS[:6]))
out = dict(seeds=seeds, base={k: base[k] for k in base}, arms={})
for arm in ARMS:
    if not all(arm in r["methods"] for r in R):
        continue
    print(f"\n=== {arm} − 1ply ===")
    rows = {}
    for k, nm, good in KEYS:
        v = [r["methods"][arm]["agg"][k] for r in R]
        d = [y - x for x, y in zip(base[k], v)]
        m, e, vd = verdict(d, good)
        rows[k] = dict(value=v, delta=d, mean=m, se=e, verdict=vd)
        print(f"  {nm:<18} {mean(base[k]):8.3f} → {mean(v):8.3f}   Δ {m:+.3f} ± {e:.3f}  {vd:4s}  "
              f"{['%+.3f' % x for x in d]}")
    tm = [r["methods"][arm]["agg"] for r in R]; tb = [r["methods"]["1ply"]["agg"] for r in R]
    rows["time"] = dict(ms_per_step=mean(t["ms_per_step"] for t in tm), ms_per_step_1ply=mean(t["ms_per_step"] for t in tb),
                        wall_s=mean(t["wall_s"] for t in tm), wall_s_1ply=mean(t["wall_s"] for t in tb),
                        h_per_step=mean(t["h_per_step"] for t in tm), h_per_step_1ply=27.0)
    t = rows["time"]
    print(f"  시간: 선택 {t['ms_per_step']:.2f} ms/스텝 (1-ply {t['ms_per_step_1ply']:.2f}, ×{t['ms_per_step']/t['ms_per_step_1ply']:.1f}) · "
          f"평가 1회 wall {t['wall_s']:.1f}s (1-ply {t['wall_s_1ply']:.1f}s) · h 평가 {t['h_per_step']:.1f} 상태/스텝")
    jr, sc, cr = rows["j_ratio"]["verdict"], rows["success_rate"]["verdict"], rows["exc_cruise"]["verdict"]
    rows["passes"] = (jr == "개선" and sc != "악화")
    rows["cruise_improves"] = (cr == "개선")
    out["arms"][arm] = rows

passing = [a for a, r in out["arms"].items() if r["passes"]]
if passing:
    best = min(passing, key=lambda a: out["arms"][a]["j_ratio"]["mean"])
    call = "채택 — 구조가 원인 → 단계 2"
    seg = ("순항 초과도 감소 — 단계 2 가 두 구간을 다 겨냥" if out["arms"][best]["cruise_improves"]
           else "순항 초과는 노이즈 — 구조 효과는 진입 구간 한정, 순항은 별도 재진단")
else:
    best = None
    call = "기각 — 1-ply 는 원인 아님 → S1-L2 기각, 원인 재진단"
    cr = [a for a, r in out["arms"].items() if r["cruise_improves"]]
    seg = (f"단 순항 초과 감소 arm: {cr} — 재진단의 출발점" if cr else "순항 초과도 전 arm 노이즈")
out.update(call=call, best=best, passing=passing, segment_reading=seg)
print(f"\n§26-2 판정: {call}" + (f" (통과 arm {passing}, 최대 감소 {best})" if best else ""))
print(f"순항 해석: {seg}")
if len(passing) == 1:
    print("⚠ 통과 arm 이 하나뿐 — 다중 비교(3 arm) 주의 (§26-2)")
if out_path:
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1, default=float)
    print("→", out_path)
