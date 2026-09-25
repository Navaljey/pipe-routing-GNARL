#!/usr/bin/env python3
"""§32-1 순위 라벨 캐시를 H2-b(obs[80:132]) 로 옮긴다 — A* 재계산 없음 (라벨 t 는 관측과 무관).

블록 점유율은 장애물 격자가 필요하므로 행마다 시나리오를 찾는다: 행의 목표 좌표 obs[3:6]×DIMS 로
후보 시나리오를 고르고, 위치 obs[0:3]×DIMS 가 그 시나리오의 자유 칸인지로 검증한다. 모호하거나 못 찾은 행이
하나라도 있으면 멈춘다. obs(학습 풀) → TRAIN_POOL, e_obs(평가 풀, 계기판 전용) → EVAL_POOL.
사용 (_ab/_cache 에서): python3 migrate_rank_cache_blocks.py <..._o80k_djfix.npz>
출력: 같은 이름의 _o132k_djfix.npz (노트북 cfg.obs_blocks=True 의 RANK_CACHE 이름)
"""
import contextlib, io, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

src = sys.argv[1]
ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
exec(compile("import os, sys, time, torch\nimport torch.nn as nn\nimport torch.nn.functional as F", "<p>", "exec"), ns)
for tag, s in load_cells(NB):
    if tag in ("3.", "4."):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(s, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":
            c = ns["cfg"]; c.pretrain_h = False; c.h_rank_batches = 0; c.h_rank_metrics = False
            c.obs_capture = True; c.obs_align_norm = "k"; c.legacy_dj_edge_bug = False; c.obs_blocks = True
assert ns["OBS_DIM"] == 132
DIMS = np.asarray(ns["DIMS"]); blocks = ns["block_channels"]
z = dict(np.load(src)); assert z["obs"].shape[1] == 80


def upgrade(O, ptr, pool, name, order=None):
    """그룹(결정 시점) 단위로 시나리오를 정한다. 같은 목표를 가진 시나리오가 풀에 몇 쌍 있어(학습 풀 4쌍)
    목표만으로는 모호한 그룹이 생긴다 — 한 시나리오의 그룹은 캐시 안에서 연속으로 놓이므로(라벨 생성 순서)
    모호한 그룹은 앞뒤의 모호하지 않은 그룹과 같은 시나리오로 정하고, 끝에 연속성·순서를 검증한다."""
    pos = np.rint(O[:, 0:3].astype(np.float64) * DIMS).astype(np.int64)
    goal = np.rint(O[:, 3:6].astype(np.float64) * DIMS).astype(np.int64)
    by_goal = {}
    for i, sc in enumerate(pool):
        by_goal.setdefault(tuple(int(v) for v in sc["goal"]), []).append(i)
    G = len(ptr) - 1; cand = []
    for g in range(G):
        a, b = ptr[g], ptr[g + 1]
        assert (goal[a:b] == goal[a]).all(), "그룹 안 목표 불일치"
        cand.append([ci for ci in by_goal.get(tuple(int(v) for v in goal[a]), [])
                     if pool[ci]["free"][pos[a:b, 0], pos[a:b, 1], pos[a:b, 2]].all()])
    assert all(len(c) >= 1 for c in cand), f"{name}: 후보가 없는 그룹이 있다"
    n_amb = sum(len(c) > 1 for c in cand)
    if order is not None:                     # 라벨 생성 순서(샤드 i::4 이어붙임)를 따라 걸으며 배정한다
        gs = np.full(G, -1); k = 0
        for g in range(G):
            while cand[g].count(order[k]) == 0:
                k += 1
                assert k < len(order), f"{name}: 순서대로 배정하지 못했다 (그룹 {g}, 후보 {cand[g]})"
            gs[g] = order[k]
            assert gs[g] in cand[g]
    else:
        assert n_amb == 0, f"{name}: 목표로 모호한 그룹 {n_amb} (순서 정보 없음)"
        gs = np.array([c[0] for c in cand])
    # 검증: 각 시나리오의 그룹은 한 덩어리로 연속이어야 한다
    runs = [gs[0]] + [gs[i] for i in range(1, G) if gs[i] != gs[i - 1]]
    assert len(runs) == len(set(runs)), f"{name}: 시나리오 그룹이 연속이 아니다"
    scen = np.repeat(gs, np.diff(ptr))
    B = np.zeros((len(O), 52), np.float32)
    for s_ in np.unique(scen):
        i = np.where(scen == s_)[0]; B[i] = blocks(pool[int(s_)]["pad"], pos[i])
    print(f"  {name}: {len(O):,}행 · 그룹 {G:,} (목표로 모호 {n_amb} → 생성 순서로 해소) · 시나리오 {len(runs)}개 · 순서 {runs[:6]}… → 132차원")
    return np.concatenate([O, B], axis=1).astype(np.float32), runs


exp = [i for w in range(4) for i in range(w, len(ns["TRAIN_POOL"]), 4)]
z["obs"], runs = upgrade(z["obs"], z["ptr"], ns["TRAIN_POOL"], "obs (학습 풀)", order=exp)
assert runs == [e for e in exp if e in set(runs)], "학습 풀 그룹 순서가 샤드 순서(i::4)와 다르다"
z["e_obs"], _ = upgrade(z["e_obs"], z["e_ptr"], ns["EVAL_POOL"], "e_obs (평가 풀 · 계기판)")
out = src.replace("_o80k_djfix", "_o132k_djfix"); assert out != src
np.savez_compressed(out, **z); print("→", out)
