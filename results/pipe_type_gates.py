#!/usr/bin/env python3
"""§36-6 관종 채널 · G0 마스크 · 흐름 뒤집기 생성기 — 건전성 관문 3개 (구현 검증, 학습 없음).

① 기본값(obs_pipe_type=False · gravity_frac=0) 이면 수정 전 노트북과 비트 단위 동일 — OBS_DIM · 캐시 이름 · 체크리스트 ·
   A* 경로 상태 마스크·obs · 평가 풀 48개 랜덤 롤아웃 (obs·마스크·보상·종단)
② obs_pipe_type=True · gravity_frac=0.5 — 관종 풀 생성 통계 · 체크리스트 12항목 · 뒤집지 않은 중력관의 G0 A* J 가
   §36 원자료(step1_slope_feasibility.json, G0) 와 일치
③ 80 → 82 zero-padding 전이 — Step 1 체크포인트(C4 2M 시드 1)를 관종 채널 켠 정책에 옮겨 압력관 평가 시나리오 행동·J 일치 ·
   중력관 t=0 · 돌연변이(환경이 gravity 를 마스크에 안 넘김) 에서 체크리스트 [11] 이 FAIL 하는지

사용 (_ab/_cache 에서):
  git show 1bda5b0:step1_train.ipynb > /tmp/old_nb.ipynb
  python3 ../../results/pipe_type_gates.py --old /tmp/old_nb.ipynb --out ../../results/step1_pipe_type_gates.json
"""
import argparse, contextlib, io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, ARMS, NB  # noqa: E402

BASE = dict(ARMS["L"]); BASE.update(h_rank_batches=256, obs_align_norm="k", obs_capture=True)


def build(nb, over=None, policy_cls=False):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec("import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F", ns)
    log = io.StringIO()
    for tag, src in load_cells(nb):
        if tag in ("3.", "4.", "5."):
            with contextlib.redirect_stdout(log):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                c = ns["cfg"]; c.pretrain_h = False; c.bc_batches = 0
                for k, v in (over or {}).items():
                    setattr(c, k, v)
        if tag == "8." and policy_cls:     # 정책 클래스 정의만 (모델 생성·학습 전까지)
            exec(compile(src.split("from stable_baselines3.common.monitor import Monitor")[0], "<nb:8a>", "exec"), ns)
    return ns, log.getvalue()


def checklist_lines(log):
    return [l.strip() for l in log.splitlines() if l.strip().startswith(("PASS", "FAIL"))]


def gate1(old_nb):
    old, lo = build(old_nb); new, ln = build(NB)
    n = bad_m = bad_o = 0
    for pool in ("TRAIN_POOL", "EVAL_POOL"):
        for so, sn in zip(old[pool], new[pool]):
            assert so["start"] == sn["start"] and so["dirs"] == sn["dirs"]
            for sol in so["astar"].values():
                for (pos, dd, ss, gg) in sol["states"][::3]:
                    n += 1
                    bad_m += not np.array_equal(old["action_mask_27"](so["pad"], pos, dd, ss, so["goal"]),
                                                new["action_mask_27"](sn["pad"], pos, dd, ss, sn["goal"]))
                    bad_o += not np.array_equal(old["build_obs"](so["pad"], pos, dd, ss, gg, so["goal"]),
                                                new["build_obs"](sn["pad"], pos, dd, ss, gg, sn["goal"]))
    eo = old["PipeRoutingEnvStep1"](old["EVAL_POOL"], eval_mode=True)
    en = new["PipeRoutingEnvStep1"](new["EVAL_POOL"], eval_mode=True)
    diff = 0
    for i in range(len(old["EVAL_POOL"])):
        a, _ = eo.reset(options={"index": i}); b, _ = en.reset(options={"index": i})
        diff += not np.array_equal(a, b)
        r = np.random.default_rng(i)
        for _ in range(60):
            m1, m2 = eo.action_masks(), en.action_masks(); diff += not np.array_equal(m1, m2)
            act = int(r.choice(np.where(m1)[0]))
            a, r1, t1, u1, _ = eo.step(act); b, r2, t2, u2, _ = en.step(act)
            diff += (not np.array_equal(a, b)) or r1 != r2 or t1 != t2 or u1 != u2
            if t1 or u1:
                break
    return dict(obs_dim=[old["OBS_DIM"], new["OBS_DIM"]], cache_same=old["CACHE"] == new["CACHE"],
                checklist_same=checklist_lines(lo) == checklist_lines(ln), n_states=n,
                mask_diff=int(bad_m), obs_diff=int(bad_o), rollout_diff=int(diff))


def gate2():
    ns, log = build(NB, dict(BASE, obs_pipe_type=True, gravity_frac=0.5))
    sf = json.load(open(os.path.join(HERE, "step1_slope_feasibility.json")))
    ref = {(r["pool"], r["idx"]): r for r in sf["rows"]}
    nchk = bad = nsw = 0
    for pname, pool in (("train", ns["TRAIN_POOL"]), ("eval", ns["EVAL_POOL"])):
        for i, sc in enumerate(pool):
            if not sc.get("gravity"):
                continue
            if sc["swapped"]:
                nsw += 1; continue
            r = ref[(pname, i)]["orig"]["G0"]
            if r["status"] == "ok":
                nchk += 1; bad += abs(sc["astar"][sc["dirs"][0]]["J"] - r["J"]) > 1e-9
    return dict(obs_dim=ns["OBS_DIM"], gnarl_obs_dim=ns["GNARL_OBS_DIM"], cache=ns["CACHE"],
                pipe_type_stats=ns["PIPE_TYPE_STATS"], checklist=checklist_lines(log),
                g0_J_vs_s36_checked=nchk, g0_J_vs_s36_mismatch=int(bad), swapped=nsw)


def gate3(ckpt):
    import torch
    from gymnasium import spaces
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    ns80, _ = build(NB, BASE)
    ns82, _ = build(NB, dict(BASE, obs_pipe_type=True, gravity_frac=0.5), policy_cls=True)
    m80 = MaskablePPO.load(ckpt, device="cpu", print_system_info=False)
    sd = ns82["zero_pad_state_dict"](m80.policy.state_dict(), ns82["OBS_DIM"])
    pol = ns82["TrueGNARLPolicy"](spaces.Box(-1, 1, (ns82["GNARL_OBS_DIM"],), np.float32), spaces.Discrete(27),
                                  lambda _: 1e-4, net_arch=dict(pi=[64], vf=[64]))
    own = pol.state_dict()
    skip = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    pol.load_state_dict({k: v for k, v in sd.items() if k not in skip}, strict=False)

    def act(p, obs, m):
        with torch.no_grad():
            return int(p.get_distribution(torch.as_tensor(obs[None]), action_masks=m[None]).get_actions(deterministic=True)[0])
    e80 = ns80["PipeRoutingEnvStep1"](ns80["EVAL_POOL"], eval_mode=True)
    e82 = ns82["PipeRoutingEnvStep1"](ns82["EVAL_POOL"], eval_mode=True)
    same = n_p = 0; grav = []
    for i, sc in enumerate(ns82["EVAL_POOL"]):
        o2, _ = e82.reset(options={"index": i})
        if not sc["gravity"]:
            n_p += 1
            o1, _ = e80.reset(options={"index": i}); s1, s2 = [], []
            for _ in range(400):
                a1 = act(m80.policy, o1, e80.action_masks()); a2 = act(pol, o2, e82.action_masks())
                s1.append(a1); s2.append(a2)
                o1, _, d1, u1, i1 = e80.step(a1); o2, _, d2, u2, i2 = e82.step(a2)
                if d1 or u1 or d2 or u2:
                    break
            same += (s1 == s2) and abs(i1["j_agent"] - i2["j_agent"]) < 1e-9
        else:
            for _ in range(400):
                o2, _, d2, u2, i2 = e82.step(act(pol, o2, e82.action_masks()))
                if d2 or u2:
                    break
            grav.append(dict(success=i2["success"], deadlock=i2["deadlock"], timeout=i2["timeout"], j_ratio=i2["j_ratio"]))
    E = ns82["PipeRoutingEnvStep1"]; orig = E.action_masks
    E.action_masks = lambda self: ns82["action_mask_27"](self.pad, self.pos, self.dir, self.s, self.goal, self.k)
    mut = ns82["run_checklist_tests"](ns82["EVAL_POOL"])
    E.action_masks = orig
    g = {k: float(np.nanmean([x[k] for x in grav])) for k in grav[0]}
    return dict(ckpt=ckpt, skipped_keys=skip, pressure_same=int(same), pressure_n=n_p,
                gravity_t0=dict(n=len(grav), **g),
                mutation_gate11=bool(mut["[11] 중력관 역구배 금지 (G0) · 환경=A* 마스크"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="수정 전 노트북 (git show 1bda5b0:step1_train.ipynb)")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "_ab", "L_2000k_bc_s1", "checkpoint_step1_final.zip"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = dict(gate1=gate1(a.old)); print(out["gate1"], flush=True)
    out["gate2"] = gate2(); print({k: v for k, v in out["gate2"].items() if k != "checklist"}, flush=True)
    out["gate3"] = gate3(os.path.abspath(a.ckpt)); print(out["gate3"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
