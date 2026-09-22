#!/usr/bin/env python3
"""2M 모델의 실패를 분해한다 — 관측을 바꾸기 전에 '무엇을 몰라서 실패하는가' 를 좁힌다.

노트북 셀(3·4·5)을 그대로 exec 해 환경을 세우고, 체크포인트를 결정적(argmin f)으로
평가 풀 48개에 돌리면서 궤적을 통째로 기록한다. 집계는 세 가지 질문에 맞춘다:

  ① 실패는 어떤 종류인가        — deadlock / timeout / 기타
  ② 실패가 어디서 나는가        — 목표까지 얼마나 접근했는가(d_min), 맴도는가(재방문)
  ③ length_ratio 는 어디서 붙는가 — 낭비 스텝을 '목표까지 거리 × 국소 혼잡도 × 수직여부' 로 쪼갠다

사용: python3 diagnose_failures.py <런 디렉터리> [...] [--out out.json]
"""
import argparse, contextlib, io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB, ARMS  # noqa: E402

NEAR_GOAL = 8          # '목표 근처' 기준 (맨해튼 칸)
OCC_R = 3              # 국소 혼잡도 반경
OCC_HI = 0.25          # '장애물 근처' 기준 (반경 3 안 막힌 비율)


def build_ns(override):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in ("3.", "4.", "5."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                for k, v in override.items():
                    setattr(ns["cfg"], k, v)
                ns["cfg"].pretrain_h = False
    return ns


def occupancy(pad, pos, PAD, r=OCC_R):
    """pos 둘레 (2r+1)^3 에서 막힌 칸의 비율. pad 는 자유=True."""
    x, y, z = (int(pos[0]) + PAD, int(pos[1]) + PAD, int(pos[2]) + PAD)
    blk = pad[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1]
    return float(1.0 - blk.mean()) if blk.size else float("nan")


def run_episodes(ns, model):
    import torch
    env = ns["PipeRoutingEnvStep1"](ns["EVAL_POOL"], eval_mode=True, seed=0)
    PAD, DIRS26, STAY = ns["PAD"], ns["DIRS26"], ns["STAY"]
    rows = []
    for i in range(len(ns["EVAL_POOL"])):
        obs, _ = env.reset(options={"index": i})
        sc = env.sc
        goal = np.asarray(env.goal, dtype=int)
        d = lambda p: int(np.abs(np.asarray(p, dtype=int) - goal).sum())
        traj, occs, dists, vert = [tuple(env.pos)], [occupancy(env.pad, env.pos, PAD)], [d(env.pos)], []
        done, info = False, {}
        while not done:
            m = env.action_masks()
            with torch.no_grad():
                ot = torch.as_tensor(obs[None], device=model.device)
                dist = model.policy.get_distribution(ot, action_masks=m[None])
                a = int(dist.get_actions(deterministic=True)[0])
            vert.append(0 if a == STAY else int(DIRS26[a][2] != 0))
            obs, r, term, trunc, info = env.step(a)
            traj.append(tuple(env.pos)); occs.append(occupancy(env.pad, env.pos, PAD)); dists.append(d(env.pos))
            done = term or trunc
        base = env.base
        dists = np.asarray(dists); occs = np.asarray(occs); vert = np.asarray(vert)
        prog = np.diff(dists)                      # <0 이면 목표에 가까워진 스텝
        waste = prog >= 0                          # 진전이 없는 스텝 = '낭비'
        rows.append(dict(
            idx=i, success=info["success"], deadlock=info["deadlock"], timeout=info["timeout"],
            steps=len(traj) - 1, d0=int(dists[0]), d_min=int(dists.min()), d_end=int(dists[-1]),
            revisit=len(traj) - len(set(traj)),
            cells_last100=len(set(traj[-100:])),
            near_goal_steps=int((dists[1:] <= NEAR_GOAL).sum()),
            waste=int(waste.sum()),
            waste_near=int((waste & (dists[1:] <= NEAR_GOAL)).sum()),
            waste_obst=int((waste & (occs[1:] >= OCC_HI)).sum()),
            waste_vert=int((waste & (vert == 1)).sum()),
            vert_steps=int(vert.sum()),
            occ_mean=float(occs.mean()),
            length_ratio=info["length_ratio"], n_bends=info["n_bends"], bends_base=info["bends_base"],
            astar_cells=float(base["n_cells"]) if base else float("nan"),
            astar_occ=float(np.mean([occupancy(sc["pad"], p, PAD) for (p, _, _, _) in base["states"]]))
            if base else float("nan"),
            astar_vert=float(np.mean([int(b[0][2] != a_[0][2]) for a_, b in
                                      zip(base["states"][:-1], base["states"][1:])])) if base else float("nan"),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "step1_failure_diag.json"))
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    override = dict(ARMS["L"])
    for kv in a.set:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except Exception: pass
        override[k] = v

    from sb3_contrib import MaskablePPO
    ns = build_ns(override)
    allrows = {}
    for d in a.dirs:
        f = os.path.join(d, "checkpoint_step1_final.zip")
        if not os.path.exists(f):
            print("없음:", f); continue
        model = MaskablePPO.load(f, device="cpu", print_system_info=False)
        rows = run_episodes(ns, model)
        allrows[os.path.basename(d.rstrip("/"))] = rows
        print(f"{os.path.basename(d.rstrip('/'))}: {len(rows)}개 에피소드", flush=True)
    json.dump(allrows, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
    print("→", a.out)


if __name__ == "__main__":
    main()
