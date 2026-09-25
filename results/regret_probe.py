#!/usr/bin/env python3
"""§26-4 원인 재진단 — 초과 J 를 **결정 하나하나의 후회(regret)** 로 분해한다. 학습·모델 변경 없음.

정책(1-ply, 결정적)을 평가 풀에 돌리며, 유효 이동이 2개 이상인 결정 시점마다 모든 유효 이웃 s'_i 에서
k 제약 A* 로 남은 최적 J y_i 를 구한다 (평가 풀 — 계기판 용도, 학습 라벨 아님 §1).
  t_i = ΔJ_i + y_i            (참 비용-to-go, kg)
  regret = t_chosen − min_i t_i ≥ 0
⚠ ΔJ 는 환경이 실제로 청구하는 `move_J` 로 계산한다. 노트북의 `move_J_all`(정책 관측의 ΔJ)은 √2 방향 12개의
직진에 45° 엘보(2.09 kg)를 잘못 붙인다 (TURN_DEG 부동소수점 — §26-4). q 는 정책이 실제로 보는 값(move_J_all)
그대로 두고, 그 오차를 뺀 q_fix 도 함께 기록한다.

A* 가 J-최적이므로 y(s_t) = min_i t_i (Bellman) 이고, 궤적 전체의 regret 합은 **정확히** 초과 J 가 된다:
  Σ regret = J_agent − y(s_0) = J_agent − 기준선 J.     ← 이 항등식을 건전성 검사로 쓴다.

regret 을 구간(순항/진입, §24-6)·A* 경로 위/밖·실수 종류로 나누고, 정책이 실제로 방문한 결정 시점에서
f 순위(q_i = ΔJ_i + h(s'_i))가 참 순위(t_i)와 얼마나 맞는지(on-policy top-1)를 잰다.

사용 (_ab/_cache 에서): python3 regret_probe.py ../L_300k_h3_s1 --out ../../results/_rp/h3_s1.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from regrade import build_ns          # noqa: E402
from run_step1_ab import ARMS         # noqa: E402

APPROACH_ZONE = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="checkpoint_step1_final")
    ap.add_argument("--blocks", action="store_true", help="H2-b 체크포인트 (obs_blocks=True, §32)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    override = dict(ARMS["L"]); override.update(h_rank_batches=256, obs_align_norm="k", obs_capture=True,
                                              obs_blocks=bool(a.blocks))
    from sb3_contrib import MaskablePPO
    ns = build_ns(override)
    model = MaskablePPO.load(os.path.join(a.run_dir, a.ckpt + ".zip"), device="cpu", print_system_info=False)
    pol = model.policy
    JM, DIRS, STAY = float(ns["J_MAX"]), ns["DIRS_ARR"], ns["STAY"]
    astar, mask27, mJall, bob = ns["astar_gnarl"], ns["action_mask_27"], ns["move_J_all"], ns["build_obs_batch"]
    move_J, SPEC, MOVE_DIST = ns["move_J"], ns["SPEC"], ns["MOVE_DIST"]
    TURN_DEG = ns["TURN_DEG"]
    env = ns["PipeRoutingEnvStep1"](ns["EVAL_POOL"], eval_mode=True)
    episodes, t0 = [], time.time()
    n_astar = n_fail = 0
    for ei in range(len(ns["EVAL_POOL"])):
        obs, _ = env.reset(options={"index": ei})
        goal = tuple(int(v) for v in env.goal); goal_a = np.asarray(goal)
        pad, k = env.pad, env.k
        base = env.base
        on_path = set((tuple(p), int(d_), int(s_)) for (p, d_, s_, _) in base["states"]) if base else set()
        memo = {}

        def y_of(st):
            nonlocal n_astar, n_fail
            if st[0] == goal:
                return 0.0
            if st not in memo:
                n_astar += 1
                sol = astar(pad, st[0], goal, st[1], s0=st[2])
                if sol is None:
                    n_fail += 1
                memo[st] = float(sol["J"]) if sol is not None else float("nan")
            return memo[st]

        dists = [int(np.abs(np.asarray(env.pos) - goal_a).sum())]
        decs, done, info, gJ = [], False, {}, [0.0]
        y0 = y_of((tuple(int(v) for v in env.pos), int(env.dir), int(env.s)))
        while not done:
            pos, d, s = tuple(int(v) for v in env.pos), int(env.dir), int(env.s)
            m = env.action_masks()
            with torch.no_grad():
                dist = pol.get_distribution(torch.as_tensor(obs[None]), action_masks=m[None])
                act = int(dist.get_actions(deterministic=True)[0])
            acts = np.where(m[:26])[0]
            if len(acts) >= 2:
                dJp = mJall(d)                                           # 정책이 보는 ΔJ (버그 포함)
                dJ = np.array([move_J(i, d, SPEC)[0] for i in range(26)])  # 환경이 청구하는 참 ΔJ
                kids = []
                for i in acts:
                    i = int(i)
                    npos = (pos[0] + int(DIRS[i, 0]), pos[1] + int(DIRS[i, 1]), pos[2] + int(DIRS[i, 2]))
                    kids.append((i, (npos, i, min(s + 1, k) if i == d else 1)))
                t = np.array([float(dJ[i]) + y_of(st) for i, st in kids])
                P = np.array([st[0] for _, st in kids]); D = np.array([st[1] for _, st in kids])
                S = np.array([st[2] for _, st in kids]); G = env.g_J + np.array([float(dJ[i]) for i, _ in kids])
                with torch.no_grad():
                    h = pol.h_admissible(torch.as_tensor(bob(pad, P, D, S, G, goal, k))).numpy()[:, 0] * JM
                q = np.array([float(dJp[i]) for i, _ in kids]) + h
                q_fix = np.array([float(dJ[i]) for i, _ in kids]) + h
                ids = [i for i, _ in kids]
                c = ids.index(act)
                best = int(np.nanargmin(t))
                decs.append(dict(step=len(gJ) - 1, n=len(ids), regret=float(t[c] - np.nanmin(t)),
                                 chosen_turn=int(act != d), best_turn=int(ids[best] != d),
                                 same_as_best=int(c == best), on_path=int((pos, d, s) in on_path),
                                 top1=int(int(np.argmin(q)) == best),
                                 top1_fix=int(int(np.argmin(q_fix)) == best),
                                 edge_dir=int(abs(float(MOVE_DIST[d]) - 2 ** .5) < 1e-6),
                                 best_straight=int(ids[best] == d), chosen_straight=int(act == d),
                                 # 선택한 수와 최선 수의 참 비용 차이를 h 가 얼마나 틀리게 봤는가 (kg)
                                 q_gap=float(q[c] - q[best]), t_gap=float(t[c] - t[best]),
                                 turn_deg_chosen=float(TURN_DEG[d, act]) if act != d else 0.0,
                                 nan=int(np.isnan(t).any())))
            obs, r, term, trunc, info = env.step(act)
            dists.append(int(np.abs(np.asarray(env.pos) - goal_a).sum())); gJ.append(float(env.g_J))
            done = term or trunc
        dd = np.asarray(dists); inz = dd <= APPROACH_ZONE
        ent = int(np.argmax(inz)) if inz.any() else len(dd) - 1
        for x in decs:
            x["seg"] = "cruise" if x["step"] < ent else "approach"
        yT = 0.0 if info["success"] else y_of((tuple(int(v) for v in env.pos), int(env.dir), int(env.s)))
        episodes.append(dict(idx=ei, success=info["success"], j_agent=float(env.g_J),
                             j_base=float(base["J"]) if base else float("nan"), y0=y0, yT=yT,
                             regret_sum=float(sum(x["regret"] for x in decs)), decisions=decs))
        print(f"  ep {ei:2d} succ {int(info['success'])} excess {env.g_J - (base['J'] if base else 0):6.2f} "
              f"Σregret {episodes[-1]['regret_sum']:6.2f} dec {len(decs):3d} | A* {n_astar} ({n_fail} 실패) "
              f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(run=os.path.basename(a.run_dir.rstrip("/")), n_astar=n_astar, n_astar_fail=n_fail,
                   episodes=episodes), open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
    print("→", a.out)


if __name__ == "__main__":
    main()
