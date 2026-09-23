#!/usr/bin/env python3
"""'전체 평가 에피소드 기준 J_ratio' 를 §2 종단 원칙대로 계산해 본다.

§2 / fail_cost_norm 주석: **미완성 경로의 J = 쓴 J + 남은 최소 J** (중간에 끊었다고 깎아주지 않는다).
그대로 적용하면 실패 에피소드의 J_ratio = (g_J + 종단상태에서의 A* 남은 최적 J) / 기준선 J 다.
이렇게 두어야 '일찍 포기할수록 점수가 좋아지는' 역전이 생기지 않는다.

사용: python3 jratio_all_probe.py <런 디렉터리> [...] [--set k=v]
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
                ns["cfg"].pretrain_h = False
    return ns


def run(ns, model):
    import torch
    env = ns["PipeRoutingEnvStep1"](ns["EVAL_POOL"], eval_mode=True, seed=0)
    astar, h_lb = ns["astar_gnarl"], ns["h_octile3d_J"]
    rows = []
    for i in range(len(ns["EVAL_POOL"])):
        obs, _ = env.reset(options={"index": i})
        sc, base = env.sc, env.base
        done, info = False, {}
        while not done:
            m = env.action_masks()
            with torch.no_grad():
                d = model.policy.get_distribution(torch.as_tensor(obs[None], device=model.device),
                                                  action_masks=m[None])
                a = int(d.get_actions(deterministic=True)[0])
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        owed, exact = 0.0, True
        if info["success"] < 0.5:
            sol = astar(env.pad, tuple(env.pos), tuple(int(v) for v in env.goal), int(env.dir), s0=int(env.s))
            if sol is not None:
                owed = float(sol["J"])
            else:                                   # 그 상태에서 도달 자체가 불가능 → 하한으로
                owed, exact = float(h_lb(env.pos, env.goal)), False
        rows.append(dict(idx=i, success=info["success"], deadlock=info["deadlock"],
                         j_agent=float(env.g_J), j_base=float(base["J"]),
                         len_agent=float(env.len_m), len_base=float(base["len_m"]),
                         bends=float(env.b90 + env.b45),
                         bends_base=float(base["bends90"] + base["bends45"]),
                         owed=owed, owed_exact=exact))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(HERE, "step1_jratio_all.json"))
    a = ap.parse_args()
    override = dict(ARMS["L"])
    for kv in a.set:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except Exception: pass
        override[k] = v
    from sb3_contrib import MaskablePPO
    ns = build_ns(override)
    out = {}
    for d in a.dirs:
        f = os.path.join(d, "checkpoint_step1_final.zip")
        if not os.path.exists(f):
            print("없음:", f); continue
        model = MaskablePPO.load(f, device="cpu", print_system_info=False)
        out[os.path.basename(d.rstrip("/"))] = run(ns, model)
        print(f"{os.path.basename(d.rstrip('/'))} 완료", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
    print("→", a.out)


if __name__ == "__main__":
    main()
