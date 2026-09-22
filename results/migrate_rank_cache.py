#!/usr/bin/env python3
"""순위 라벨 캐시를 새 관측 차원으로 옮긴다 — A* 를 다시 돌리지 않는다.

비싼 것은 라벨(t)이고 그것은 관측과 무관하다. 새로 붙는 obs[76:78] 은
(위치·목표·방향)만으로 정해지는데 셋 다 기존 obs 에서 **정확히 복원된다**:
  pos = round(obs[0:3] × DIMS) · goal = round(obs[3:6] × DIMS) · dir = argmax(obs[6:32])
따라서 32분짜리 재생성 대신 몇 초짜리 변환으로 끝난다.

사용: python3 migrate_rank_cache.py <옛 캐시.npz> --norm k --out-dir <디렉터리>
"""
import argparse, contextlib, io, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--norm", default="k", choices=["k", "diag"])
ap.add_argument("--out-dir", default=".")
a = ap.parse_args()

ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
exec(compile("import os, sys, time", "<p>", "exec"), ns)
for tag, src in load_cells(NB):
    if tag in ("3.", "4."):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":
            ns["cfg"].pretrain_h = False
            ns["cfg"].obs_align_norm = a.norm
OBS_DIM, DIMS, align = ns["OBS_DIM"], ns["DIMS"], ns["align_channels"]

z = dict(np.load(a.src))
old_w = z["obs"].shape[1]
assert old_w == OBS_DIM - 2, f"이 변환은 {OBS_DIM-2} → {OBS_DIM} 전용이다 (현재 {old_w})"

def upgrade(obs):
    pos = np.rint(obs[:, 0:3].astype(np.float64) * DIMS)
    goal = np.rint(obs[:, 3:6].astype(np.float64) * DIMS)
    dirs = obs[:, 6:32].argmax(axis=1)
    assert np.abs(obs[:, 0:3] * 1.0 - (pos / DIMS)).max() < 1e-5, "위치 복원 실패"
    assert obs[np.arange(len(obs)), 6 + dirs].min() == 1.0, "방향 복원 실패"
    return np.concatenate([obs, align(pos, dirs, goal)], axis=1).astype(np.float32)

for key in ("obs", "e_obs"):
    z[key] = upgrade(z[key])
    print(f"  {key}: {old_w} → {z[key].shape[1]}차원 ({len(z[key]):,}행)")

base = os.path.basename(a.src).replace(".npz", "")
for suf in (f"_o{old_w}", f"_o{OBS_DIM}k", f"_o{OBS_DIM}diag"):
    base = base.replace(suf, "")
out = os.path.join(a.out_dir, f"{base}_o{OBS_DIM}{a.norm}.npz")
np.savez_compressed(out, **z)
print("→", out)
