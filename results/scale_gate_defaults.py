#!/usr/bin/env python3
"""§42 — 오늘 노트북 변경(장애물 파라미터화 · 비트 패킹 · A* 휴리스틱 선택 · J_MAX 고정 옵션 · scaled fail_cost · 라벨량 옵션)이
기본값에서 수정 전 노트북과 비트 단위 동일한지. pipe_type_gates.gate1 (OBS_DIM · 캐시 이름 · 체크리스트 · A* 경로 상태 마스크·obs ·
평가 48개 랜덤 롤아웃) + A* J · 전개 수 (평가 풀 전 시나리오 · 시작 방향 전부) + J_MAX · fail_cost 값.
사용 (_ab/_cache 에서): git show f2e69d5:step1_train.ipynb > /tmp/old.ipynb ; python3 ../../results/scale_gate_defaults.py --old /tmp/old.ipynb --out ../../results/step1_scale_gate_defaults.json"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pipe_type_gates import build, gate1  # noqa: E402
from run_step1_ab import NB  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--old", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = dict(gate1=gate1(a.old))
    old, _ = build(a.old); new, _ = build(NB)
    bad = n = 0
    for so, sn in zip(old["EVAL_POOL"], new["EVAL_POOL"]):
        for d in so["dirs"]:
            ro = old["astar_gnarl"](so["pad"], so["start"], so["goal"], d); rn = new["astar_gnarl"](sn["pad"], sn["start"], sn["goal"], d)
            n += 1; bad += not (ro["J"] == rn["J"] and ro["expanded"] == rn["expanded"] and ro["states"] == rn["states"])
    out.update(astar_checked=n, astar_diff=bad, J_MAX=[old["J_MAX"], new["J_MAX"]],
               fail_cost=[old["fail_cost_norm"]((0, 0, 0), (5, 5, 5)), new["fail_cost_norm"]((0, 0, 0), (5, 5, 5))],
               pack=bool(new["PACK_SCENARIOS"]), heuristic=new["cfg"].astar_heuristic)
    json.dump(out, open(a.out, "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))

if __name__ == "__main__":
    main()
