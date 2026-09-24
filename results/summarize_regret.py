#!/usr/bin/env python3
"""§26-4 regret 분해 요약 — results/_rp/*.json (regret_probe.py 출력) → 시드 평균 ± SE.

성공 에피소드만 쓴다 (초과 J = Σ regret 이 정확히 성립하는지 먼저 검사).
사용: python3 summarize_regret.py "results/_rp/h3_s*.json" [--out results/step1_regret_h3.json]
"""
import glob, json, sys
from statistics import mean, stdev

files = sorted(glob.glob(sys.argv[1]))
se = lambda v: stdev(v) / len(v) ** .5 if len(v) > 1 else float("nan")
per_seed = []
for f in files:
    d = json.load(open(f))
    E = [e for e in d["episodes"] if e["success"] > 0.5]
    ident = max(abs((e["j_agent"] - e["j_base"]) - e["regret_sum"]) for e in E)
    D = [x for e in E for x in e["decisions"]]
    n_ep = len(E)
    tot = sum(x["regret"] for x in D)
    r = dict(run=d["run"], n_ep=n_ep, identity_max_err=ident, excess_kg=tot / n_ep)
    for seg in ("cruise", "approach"):
        S = [x for x in D if x["seg"] == seg]
        M = [x for x in S if x["regret"] > 1e-9]
        r[f"{seg}_kg"] = sum(x["regret"] for x in S) / n_ep
        r[f"{seg}_dec_per_ep"] = len(S) / n_ep
        r[f"{seg}_mistakes_per_ep"] = len(M) / n_ep
        r[f"{seg}_mistake_rate"] = len(M) / max(len(S), 1)
        r[f"{seg}_top1"] = mean(x["top1"] for x in S) if S else float("nan")
        r[f"{seg}_top1_fix"] = mean(x["top1_fix"] for x in S) if S else float("nan")
        r[f"{seg}_kg_per_mistake"] = sum(x["regret"] for x in M) / max(len(M), 1)
    # A* 경로 위/밖 — 정책이 기준선 경로 위에 있을 때와 벗어났을 때
    for tag, sel in (("onpath", 1), ("offpath", 0)):
        S = [x for x in D if x["on_path"] == sel]
        r[f"{tag}_kg"] = sum(x["regret"] for x in S) / n_ep
        r[f"{tag}_top1"] = mean(x["top1"] for x in S) if S else float("nan")
        r[f"{tag}_dec_share"] = len(S) / max(len(D), 1)
    # 실수 종류
    M = [x for x in D if x["regret"] > 1e-9]
    kinds = {
        "turned_should_straight": lambda x: x["chosen_turn"] and not x["best_turn"],
        "straight_should_turn": lambda x: (not x["chosen_turn"]) and x["best_turn"],
        "wrong_turn_dir": lambda x: x["chosen_turn"] and x["best_turn"],
    }
    for nm, fn in kinds.items():
        r[f"kind_{nm}_kg"] = sum(x["regret"] for x in M if fn(x)) / n_ep
        r[f"kind_{nm}_n"] = sum(1 for x in M if fn(x)) / n_ep
    # √2 방향 직진 ΔJ 버그와 맞물린 실수: 진행 방향이 √2 방향이고 최선이 직진인데 꺾었다
    B = [x for x in M if x["edge_dir"] and x["best_straight"] and not x["chosen_straight"]]
    r["bug_pattern_kg"] = sum(x["regret"] for x in B) / n_ep
    r["bug_pattern_n"] = len(B) / n_ep
    r["bug_pattern_cruise_kg"] = sum(x["regret"] for x in B if x["seg"] == "cruise") / n_ep
    # 같은 h 로 ΔJ 만 고치면 사라지는 실수 (top1 이 틀렸는데 top1_fix 는 맞는 결정) — 반사실 추정, 상한 아님
    F = [x for x in M if x["top1_fix"] and not x["top1"]]
    r["fixable_by_dJ_kg"] = sum(x["regret"] for x in F) / n_ep
    r["fixable_by_dJ_n"] = len(F) / n_ep
    per_seed.append(r)

keys = [k for k in per_seed[0] if k not in ("run",)]
agg = {k: (mean(r[k] for r in per_seed), se([r[k] for r in per_seed])) for k in keys}
print(f"시드 {[r['run'] for r in per_seed]} · 성공 에피소드 {[r['n_ep'] for r in per_seed]}")
print(f"항등식 Σregret = 초과 J  최대 오차 {max(r['identity_max_err'] for r in per_seed):.2e} kg")
def show(k, nm, fmt="{:.2f}"):
    m, e = agg[k]; print(f"  {nm:<44} {fmt.format(m)} ± {fmt.format(e)}")
print("\n[초과 J 분해 — 에피소드당 kg]")
show("excess_kg", "초과 J = Σ regret")
show("cruise_kg", "  순항 구간 regret"); show("approach_kg", "  진입 구간 regret")
show("onpath_kg", "  A* 경로 위 결정의 regret"); show("offpath_kg", "  A* 경로 밖 결정의 regret")
print("\n[결정 품질]")
for seg in ("cruise", "approach"):
    show(f"{seg}_dec_per_ep", f"{seg} 결정 시점 / 에피소드", "{:.1f}")
    show(f"{seg}_mistakes_per_ep", f"{seg} 실수 (regret>0) / 에피소드", "{:.2f}")
    show(f"{seg}_mistake_rate", f"{seg} 실수율", "{:.3f}")
    show(f"{seg}_kg_per_mistake", f"{seg} 실수 1회당 kg")
    show(f"{seg}_top1", f"{seg} on-policy top-1 (정책 관점 f)", "{:.3f}")
    show(f"{seg}_top1_fix", f"{seg} on-policy top-1 (ΔJ 버그만 고친 f)", "{:.3f}")
show("onpath_top1", "A* 경로 위 결정의 top-1", "{:.3f}"); show("offpath_top1", "A* 경로 밖 결정의 top-1", "{:.3f}")
show("offpath_dec_share", "결정 중 A* 경로 밖 비율", "{:.3f}")
print("\n[실수 종류 — 에피소드당 kg (횟수)]")
for nm in ("turned_should_straight", "straight_should_turn", "wrong_turn_dir"):
    m, e = agg[f"kind_{nm}_kg"]; n, _ = agg[f"kind_{nm}_n"]
    print(f"  {nm:<44} {m:.2f} ± {e:.2f}  ({n:.2f}회)")
print("\n[√2 방향 직진 ΔJ 버그 (§26-4)]")
show("bug_pattern_kg", "√2 방향 진행 중 직진이 최선인데 꺾은 실수 kg"); show("bug_pattern_n", "  횟수 / 에피소드")
show("bug_pattern_cruise_kg", "  그중 순항 구간 kg")
show("fixable_by_dJ_kg", "같은 h 로 ΔJ 만 고치면 최선을 고르는 실수 kg"); show("fixable_by_dJ_n", "  횟수 / 에피소드")
if "--out" in sys.argv:
    json.dump(dict(per_seed=per_seed, agg=agg), open(sys.argv[sys.argv.index("--out") + 1], "w"),
              ensure_ascii=False, indent=1, default=float)
