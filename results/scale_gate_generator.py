#!/usr/bin/env python3
"""§42 ① 관문 — 파라미터화한 장애물 생성기가 Step 1 풀 548개를 비트 단위로 재현하는지 (캐시 없는 디렉터리에서 실행)."""
import sys, io, contextlib, pickle, time, numpy as np
sys.path.insert(0, "/home/user/pipe-routing-GNARL/results")
from run_step1_ab import load_cells, NB
ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
exec(compile("import os, sys, time", "<p>", "exec"), ns)
t0=time.time()
for tag, src in load_cells(NB):
    if tag in ("3.", "4."):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag=="3.":
            ns["cfg"].pretrain_h=False; ns["cfg"].bc_batches=0
print("regen", round(time.time()-t0), "s", ns["CACHE"])
new_tr, new_ev = ns["TRAIN_POOL"], ns["EVAL_POOL"]
old_tr, old_ev = pickle.load(open("/home/user/pipe-routing-GNARL/_ab/_cache/"+ns["CACHE"],"rb"))
bad=0
for a,b in zip(old_tr+old_ev, new_tr+new_ev):
    same = (np.array_equal(a["free"],b["free"]) and tuple(a["start"])==tuple(b["start"]) and tuple(a["goal"])==tuple(b["goal"])
            and list(a["dirs"])==list(b["dirs"]) and all(abs(a["astar"][d]["J"]-b["astar"][d]["J"])<1e-9 for d in a["dirs"]))
    bad += not same
print("scenarios", len(old_tr+old_ev), "mismatch", bad)
# box count needed per grid (old cap 1000 binding?)
cfg=ns["cfg"]; rng=np.random.default_rng(0); mx=0
for _ in range(200):
    free=np.ones((cfg.NX,cfg.NY,cfg.NZ),bool); target=int(cfg.obstacle_fill*free.size); filled=0; n=0
    while filled<target:
        n+=1; sx,sy,sz=int(rng.integers(2,5)),int(rng.integers(2,5)),int(rng.integers(1,5))
        x=int(rng.integers(0,cfg.NX-sx+1)); y=int(rng.integers(0,cfg.NY-sy+1)); z=int(rng.integers(0,cfg.NZ-sz+1))
        filled+=int(free[x:x+sx,y:y+sy,z:z+sz].sum()); free[x:x+sx,y:y+sy,z:z+sz]=False
    mx=max(mx,n)
print("boxes needed max over 200 grids:", mx, "cap auto:", ns["_obstacle_box_cap"](int(cfg.obstacle_fill*cfg.NX*cfg.NY*cfg.NZ)))
