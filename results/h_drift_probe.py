#!/usr/bin/env python3
"""h 드리프트가 '순위 손상'인지 '상수 편향'인지 가른다 (§19-2 (2) 후속).

softmax(-f/τ) 는 이웃 간 **차이**만 본다. h 전체가 같은 값만큼 밀려도 정책은 그대로다.
그런데 §19 가 쓰는 h MAE 는 그 편향까지 같이 센다. 그래서 세 가지를 따로 잰다:

  raw MAE      — §19 와 같은 값 (편향 포함)
  debiased MAE — 전체 평균 오차를 뺀 뒤의 MAE (= 진짜 모양 오차)
  rank         — A* 상태의 이웃 집합 안에서 h 가 매기는 순위가 실제 남은 J 순위와
                 얼마나 맞는지 (Spearman ρ, 정책이 실제로 쓰는 양)

사용: python3 h_drift_probe.py _ab/R_300k_s1 [...]        (디렉터리 → final/best)
      python3 h_drift_probe.py _ab/A_2000k_lr1e4_s1/checkpoint_step1_500k.zip [...]  (체크포인트 직접)
"""
import json, os, sys, io, contextlib, re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB, BASE_CELLS, ARMS  # noqa: E402


def build_ns(arm_override):
    import numpy as np, torch  # noqa: F401
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<prelude>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag not in ("3.", "4.", "5.", "7."):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":
            for k, v in arm_override.items():
                setattr(ns["cfg"], k, v)
            ns["cfg"].pretrain_h = False       # 프로브는 체크포인트의 h 만 본다
    return ns


def _spearman(a, b):
    """numpy 만으로 계산하는 Spearman ρ (동점은 평균 순위)."""
    import numpy as np
    def rk(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), float); r[o] = np.arange(len(v))
        # 동점 평균 순위
        vs = v[o]; i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rk(np.asarray(a, float)), rk(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def probe(ns, model):
    import numpy as np, torch
    import torch.nn.functional as F
    J_MAX, build_obs = ns["J_MAX"], ns["build_obs"]
    rows_raw, rows_deb, rhos = [], [], []
    with torch.no_grad():
        for sc in ns["EVAL_POOL"]:
            for d, sol in sc["astar"].items():
                st = sol["states"]
                X = np.stack([build_obs(sc["pad"], p, dd, ss, gg, sc["goal"])
                              for (p, dd, ss, gg) in st])
                Y = np.array([(sol["J"] - gg) / J_MAX for (_, _, _, gg) in st], dtype=np.float32)
                h = F.softplus(model.policy.h_core(torch.as_tensor(X))).squeeze(-1).numpy()
                e = h - Y
                rows_raw.append(np.abs(e)); rows_deb.append(e)
                if len(Y) > 2:
                    r = _spearman(h, Y)
                    if r == r:
                        rhos.append(r)
    raw = float(np.concatenate(rows_raw).mean()) * J_MAX
    e = np.concatenate(rows_deb)
    deb = float(np.abs(e - e.mean()).mean()) * J_MAX
    return dict(h_mae_raw_kg=raw, h_mae_debiased_kg=deb, h_bias_kg=float(e.mean()) * J_MAX,
                spearman_mean=float(np.mean(rhos)), n_traj=len(rhos))


def main():
    from sb3_contrib import MaskablePPO
    out = {}
    ns = None
    for d in sys.argv[1:]:
        m = re.match(r"([A-Z]+)_", os.path.basename(os.path.dirname(d.rstrip("/")) if d.endswith(".zip")
                                                    else d.rstrip("/")))
        arm = m.group(1) if m else "R"
        if ns is None:                      # 시나리오 풀은 한 번만 만든다 (설정이 같으므로)
            ns = build_ns(ARMS.get(arm, {}))
        if d.endswith(".zip"):
            names, base = [os.path.basename(d)[:-4]], os.path.dirname(d)
        else:
            names, base = ["checkpoint_step1_final", "checkpoint_step1_best"], d
        for name in names:
            f = os.path.join(base, name + ".zip")
            if not os.path.exists(f):
                continue
            d = base
            model = MaskablePPO.load(f, device="cpu", print_system_info=False)
            r = probe(ns, model)
            r["timesteps"] = int(model.num_timesteps)
            r["tau"] = float(model.policy.log_tau.detach().exp())
            out[f"{os.path.basename(d.rstrip('/'))}/{name}"] = r
            print(f"{os.path.basename(d.rstrip('/')):>22} {name:<26} "
                  f"steps {r['timesteps']:>8,} | tau {r['tau']:.4f} | "
                  f"raw MAE {r['h_mae_raw_kg']:>6.2f} kg | debiased {r['h_mae_debiased_kg']:>6.2f} kg | "
                  f"bias {r['h_bias_kg']:>+7.2f} kg | Spearman {r['spearman_mean']:.3f}", flush=True)
    json.dump(out, open("h_drift_probe.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
