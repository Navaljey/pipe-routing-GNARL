#!/usr/bin/env python3
"""S1-B1 (§27) — 순위 라벨 캐시의 ΔJ 버그를 A* 재계산 없이 고친다.

버그: move_J_all 이 √2 방향 12개의 **직진**에 45° 엘보를 붙였다. 캐시에 들어간 곳:
  dj (= ΔJ/J_MAX) · t (= dj + y) · 이웃 obs[33] (= min((g+ΔJ)/J_MAX, 1))
A* 로 구한 y = t − dj 는 참값이므로 그대로 두고, 버그 행에서만 45° 엘보 몫(E45/J_MAX)을 뺀다.

버그 행 식별: 결정 시점(유효 이동 ≥ 2)은 s ≥ k 에서만 생기므로 **직진 자식은 s' = k (obs[32] = 1)**,
선회 자식은 s' = 1 (obs[32] = 1/k) 이다. 그중 자식 방향이 √2 방향인 행이 버그 행이다.
쌍(pi, pj)은 고친 t 로 다시 만든다.

검증: --verify N 이면 학습 풀 시나리오 0, 4, 8, … (샤드 0 의 앞쪽) N 개를 **수정된 코드로 처음부터** 다시 라벨링해 고친 캐시의
앞쪽 결정 시점들과 비교한다 (obs · dj · t 최대 오차).

사용 (_ab/_cache 에서): python3 fix_rank_cache_dj.py <o80k 캐시.npz> [--verify 3]
"""
import argparse, contextlib, io, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--verify", type=int, default=3)
a = ap.parse_args()

ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
exec(compile("import os, sys, time, torch\nimport torch.nn as nn\nimport torch.nn.functional as F", "<p>", "exec"), ns)
for tag, src in load_cells(NB):
    if tag in ("3.", "4.", "5.", "6.", "7b."):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":
            c = ns["cfg"]
            c.pretrain_h = False; c.bc_batches = 0; c.h_rank_batches = 0; c.h_rank_metrics = False   # 7b 가 캐시를 읽지 않게
            c.legacy_dj_edge_bug = False
K, JM, MOVE_DIST, SPEC = ns["K"], float(ns["J_MAX"]), ns["MOVE_DIST"], ns["SPEC"]
E45 = float(SPEC.elbow45_kg) / JM
assert all(abs(ns["move_J_all"](d)[i] - ns["move_J"](i, d, SPEC)[0]) < 1e-9 for d in range(26) for i in range(26)), \
    "노트북 수정이 적용되지 않았다 (move_J_all ≠ move_J)"

z = dict(np.load(a.src))


def fix(prefix):
    obs, dj, t, ptr = z[prefix + "obs"].copy(), z[prefix + "dj"].copy(), z[prefix + "t"].copy(), z[prefix + "ptr"]
    child_dir = obs[:, 6:32].argmax(1)
    straight = np.isclose(obs[:, 32], 1.0)
    edge = np.isclose(MOVE_DIST[child_dir], 2 ** .5)
    bug = straight & edge
    capped = bug & np.isclose(obs[:, 33], 1.0)
    dj[bug] -= E45; t[bug] -= E45
    obs[bug & ~capped, 33] -= E45
    pi, pj = [], []
    for s0, s1 in zip(ptr[:-1], ptr[1:]):
        tt = t[s0:s1]
        ii, jj = np.nonzero(tt[:, None] < tt[None, :] - 1e-9)
        pi.append(ii + s0); pj.append(jj + s0)
    pi = np.concatenate(pi).astype(np.int64); pj = np.concatenate(pj).astype(np.int64)
    # 방향이 바뀐 쌍 수 (버그가 순위를 얼마나 뒤집었나)
    old = set(zip(z[prefix + "pi"].tolist(), z[prefix + "pj"].tolist())); new = set(zip(pi.tolist(), pj.tolist()))
    print(f"  {prefix or 'train '}: 행 {len(t):,} · 버그 행 {int(bug.sum()):,} ({bug.mean():.3%}) · g 채널 포화 {int(capped.sum())} · "
          f"쌍 {len(old):,} → {len(new):,} (사라짐 {len(old - new):,} / 새로 생김 {len(new - old):,})")
    z[prefix + "obs"], z[prefix + "dj"], z[prefix + "t"], z[prefix + "pi"], z[prefix + "pj"] = obs, dj, t, pi, pj


fix(""); fix("e_")
out = a.src.replace(".npz", "_djfix.npz")
assert out != a.src
np.savez_compressed(out, **z)
print("→", out)

if a.verify:
    rows = []
    for sc in ns["TRAIN_POOL"][0::4][:a.verify]:   # 캐시는 4샤드(i::4)를 이어 붙여 만들었다 → 앞쪽은 시나리오 0, 4, 8, …
        goal = tuple(sc["goal"])
        for d, sol in sc["astar"].items():
            for (pos, dd, ss, gg) in sol["states"]:
                if tuple(pos) == goal:
                    continue
                r = ns["_rank_rows_for_state"](sc, tuple(pos), int(dd), int(ss), float(gg))
                if r:
                    rows += r
    O = np.stack([r[0] for r in rows]); D = np.array([r[1] for r in rows]); T = np.array([r[2] for r in rows])
    n = len(rows)
    print(f"검증: 시나리오 0,4,8,… {a.verify}개 처음부터 재라벨 {n}행 vs 고친 캐시 앞 {n}행 — "
          f"obs 최대오차 {np.abs(O - z['obs'][:n]).max():.2e} · dj {np.abs(D - z['dj'][:n]).max():.2e} · "
          f"t {np.abs(T - z['t'][:n]).max():.2e}")
