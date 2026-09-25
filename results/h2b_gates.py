#!/usr/bin/env python3
"""§32-1 건전성 관문 ①② — H2-b 구현 검증 (실행 전).
 ① obs_blocks=False 일 때 새 노트북의 obs 가 수정 전 노트북과 비트 단위 동일
 ② obs_blocks=True 의 obs[80:132] 가 §31-7 사전 점검 스크립트(h2_precheck.channels)의 H2-b 값과 경로 밖 캐시 전 행에서 일치
사용 (_ab/_cache 에서): python3 h2b_gates.py <수정 전 노트북.ipynb>
"""
import contextlib, io, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402
import h2_precheck as H  # noqa: E402

def boot(nb, **over):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time, torch\nimport torch.nn as nn\nimport torch.nn.functional as F", "<p>", "exec"), ns)
    for tag, src in load_cells(nb):
        if tag in ("3.", "4."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                c = ns["cfg"]; c.pretrain_h = False; c.bc_batches = 0; c.h_rank_batches = 0; c.h_rank_metrics = False
                for k, v in over.items(): setattr(c, k, v)
    return ns

old = boot(sys.argv[1]); new0 = boot(NB); new1 = boot(NB, obs_blocks=True)
assert new0["OBS_DIM"] == 80 and new1["OBS_DIM"] == 132, (new0["OBS_DIM"], new1["OBS_DIM"])
rng = np.random.default_rng(0); pool = old["TRAIN_POOL"]; K = old["K"]; n_cmp = 0; maxdiff = 0.0
for si in rng.choice(len(pool), 40, replace=False):
    sc_o, sc_n = old["TRAIN_POOL"][si], new1["TRAIN_POOL"][si]
    idx = np.argwhere(sc_o["free"]); pick = idx[rng.choice(len(idx), 200, replace=False)]
    dirs = rng.integers(0, 26, 200); ss = rng.integers(0, K + 1, 200); gJ = rng.random(200) * 50
    a = old["build_obs_batch"](sc_o["pad"], pick, dirs, ss, gJ, sc_o["goal"])
    b = new0["build_obs_batch"](new0["TRAIN_POOL"][si]["pad"], pick, dirs, ss, gJ, sc_o["goal"])
    c = new1["build_obs_batch"](sc_n["pad"], pick, dirs, ss, gJ, sc_o["goal"])
    assert np.array_equal(a, b), "① obs_blocks=False 인데 obs 가 달라졌다"
    assert np.array_equal(a, c[:, :80]), "obs_blocks=True 의 obs[0:80] 이 달라졌다"
    one = new1["build_obs"](sc_n["pad"], tuple(pick[0]), int(dirs[0]), int(ss[0]), float(gJ[0]), sc_o["goal"])
    assert np.allclose(one, c[0]), "build_obs 와 build_obs_batch 불일치"
    n_cmp += 200
print(f"① 통과: obs_blocks=False 가 수정 전과 비트 단위 동일 ({n_cmp} 상태 · 40 시나리오), obs[0:80] 도 동일, build_obs = build_obs_batch")

# ② 사전 점검 스크립트의 H2-b 값과 비교 (경로 밖 캐시 전 행)
z = np.load(H.CACHE); obs, ptr, nA = z["obs"], z["ptr"], int(z["n_astar_rows"]); gA = int(np.searchsorted(ptr, nA))
cand = np.load("offpath_candidates.npz")["cand"]
scen_g = np.concatenate([cand[w::4][:, 0] for w in range(4)])
O = obs[nA:]; P = ptr[gA:] - nA; rows_scen = np.repeat(scen_g, np.diff(P))
DIMS = np.asarray(new1["DIMS"]); pos = np.rint(O[:, 0:3] * DIMS).astype(np.int64)
DIRS = [tuple(int(v) for v in d) for d in new1["DIRS_ARR"]]; w = [float(np.sqrt(sum(v * v for v in d))) for d in DIRS]
_, Bref, _, _ = H.channels(new1["TRAIN_POOL"], rows_scen, pos, DIRS, w)
Bnew = np.zeros_like(Bref)
for s in np.unique(rows_scen):
    i = np.where(rows_scen == s)[0]
    Bnew[i] = new1["block_channels"](new1["TRAIN_POOL"][int(s)]["pad"], pos[i])
d = float(np.abs(Bnew - Bref).max())
assert d < 1e-6, f"② 사전 점검 값과 불일치 (최대 {d})"
print(f"② 통과: 노트북 block_channels = 사전 점검 H2-b ({len(pos):,}행 · 최대 오차 {d:.1e})")
json.dump(dict(gate1_states=n_cmp, gate2_rows=int(len(pos)), gate2_maxdiff=d),
          open(os.path.join(HERE, "step1_h2b_gates.json"), "w"), indent=1)
