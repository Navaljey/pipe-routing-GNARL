#!/usr/bin/env python3
"""착지 실패의 '가장 가까이 간 순간' 에서 k 제약 A* 를 다시 풀어, 선회가 불가피했는지 가른다.

질문: 그 자리(위치·방향·직진칸수)에서 목표로 **바로** 들어갈 수 있었는가,
      아니면 k=6 제약 때문에 큰 선회가 강제됐는가.

기준: 26-연결 격자에서 칸 수의 하한은 **체비쇼프 거리**다 (대각 이동이 세 축을 한 번에 줄인다).
      A* 가 실제로 쓰는 칸 수 − 체비쇼프 거리 = **k 제약이 강제한 추가 칸수**.

사용: python3 closest_approach_probe.py <런 디렉터리> [...]
"""
import argparse, contextlib, io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB, ARMS  # noqa: E402


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
                ns["cfg"].pretrain_h = False; ns["cfg"].bc_batches = 0
    return ns


def probe(ns, model):
    import torch
    env = ns["PipeRoutingEnvStep1"](ns["EVAL_POOL"], eval_mode=True, seed=0)
    astar, K = ns["astar_gnarl"], ns["K"]
    out = []
    for i in range(len(ns["EVAL_POOL"])):
        obs, _ = env.reset(options={"index": i})
        sc = env.sc
        goal = np.asarray(env.goal, dtype=int)
        man = lambda p: int(np.abs(np.asarray(p, dtype=int) - goal).sum())
        best = (man(env.pos), tuple(env.pos), int(env.dir), int(env.s))
        done, info = False, {}
        while not done:
            m = env.action_masks()
            with torch.no_grad():
                d = model.policy.get_distribution(torch.as_tensor(obs[None], device=model.device),
                                                  action_masks=m[None])
                a = int(d.get_actions(deterministic=True)[0])
            obs, r, term, trunc, info = env.step(a)
            dm = man(env.pos)
            if dm < best[0]:
                best = (dm, tuple(env.pos), int(env.dir), int(env.s))
            done = term or trunc
        if info["success"] > 0.5:
            continue                                   # 성공은 볼 것이 없다
        dmin, pos, dr, s = best
        cheb = int(np.abs(np.asarray(pos, dtype=int) - goal).max())
        sol = astar(sc["pad"], pos, tuple(int(v) for v in goal), dr, s0=s)
        out.append(dict(idx=i, d_min_manhattan=dmin, cheb=cheb, s_at_best=s,
                        astar_cells=int(sol["n_cells"]) if sol else -1,
                        extra=(int(sol["n_cells"]) - cheb) if sol else None,
                        bends=int(sol["bends90"] + sol["bends45"]) if sol else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(HERE, "step1_closest_approach.json"))
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
        allrows[os.path.basename(d.rstrip("/"))] = probe(ns, model)
        print(f"{os.path.basename(d.rstrip('/'))}: 실패 {len(allrows[os.path.basename(d.rstrip('/'))])}개", flush=True)
    json.dump(allrows, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
    print("→", a.out)


if __name__ == "__main__":
    main()
