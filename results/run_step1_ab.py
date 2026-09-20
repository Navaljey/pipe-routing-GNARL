#!/usr/bin/env python3
"""step1_train.ipynb 의 셀을 그대로 실행해 §0 의 붕괴 대응 손잡이(A~D)를 A/B 한다.

노트북이 유일한 소스다. 이 스크립트는 코드셀을 #@title 번호로 골라 exec 할 뿐,
환경/보상/평가/정책 로직을 따로 갖고 있지 않다 (results/run_ablation.py 와 같은 방식).

§11 프로토콜: 한 조건당 여러 시드 + 산포. 시나리오 풀은 셀 4 실행 직후에 SEED 를 덮어써
**고정**되고 (풀 생성에는 노트북 기본 SEED 가 쓰인다), 정책 초기화/롤아웃만 시드별로 달라진다.

사용:
    python3 run_step1_ab.py --arm R --seeds 1 2 3 4 --steps 300000       # 1차 런 설정(대조)
    python3 run_step1_ab.py --arm D --seeds 1 2 3 4 --steps 300000       # h 완전 동결
    python3 run_step1_ab.py --arm A --seeds 1 2 3 4                      # tau 하한
    python3 run_step1_ab.py --arm AC --seeds 1 --steps 2000000 --eval-every 100000
"""
import argparse, io, json, contextlib, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "step1_train.ipynb")
BASE_CELLS = ["3.", "4.", "5.", "6.", "7.", "8."]   # 1.(pip) 1b.(Drive) 2.(wandb) 는 건너뛴다
TRAIN_CELL = "9."

# §0 / §19-4 의 대응 손잡이 조합
ARMS = {
    # 1차 정식 학습과 같은 설정 (대조군) — tau 학습 O, h 학습 O
    "R":  dict(gnarl_freeze_h=False, gnarl_learn_tau=True,  gnarl_tau_min=0.0),
    # D. h 완전 동결 (진단용 대조군)
    "D":  dict(gnarl_freeze_h=True,  gnarl_learn_tau=True,  gnarl_tau_min=0.0),
    # A. tau 하한만 (h 는 계속 학습)
    "A":  dict(gnarl_freeze_h=False, gnarl_learn_tau=True,  gnarl_tau_min=0.05),
    # B. tau 고정
    "B":  dict(gnarl_freeze_h=False, gnarl_learn_tau=False, gnarl_tau_min=0.0),
    # C. h 워밍업 동결 (unfreeze_at 은 --unfreeze-at 으로 조정)
    "C":  dict(gnarl_freeze_h=True,  gnarl_learn_tau=True,  gnarl_tau_min=0.0,
               gnarl_unfreeze_at=100_000),
    # A+D. tau 하한 + h 완전 동결
    "AD": dict(gnarl_freeze_h=True,  gnarl_learn_tau=True,  gnarl_tau_min=0.05),
    # A+C. tau 하한 + h 워밍업 동결  ← 정식 런 후보
    "AC": dict(gnarl_freeze_h=True,  gnarl_learn_tau=True,  gnarl_tau_min=0.05,
               gnarl_unfreeze_at=100_000),
    # A+B. tau 고정(0.05) + h 학습 — 하한과 고정의 차이(S1-N1)를 가른다
    "AB": dict(gnarl_freeze_h=False, gnarl_learn_tau=False, gnarl_tau_min=0.05),
}


def load_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        m = re.match(r"#@title\s+([\w.]+)", src)
        out.append((m.group(1) if m else "", src))
    return out


def run_one(arm, seed, steps, eval_every, ckpt_every, extra, quiet_base=True):
    import numpy as np, torch
    cells = load_cells(NB)
    override = dict(ARMS[arm]); override.update(extra)
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    # 셀 1b(Drive) · 2(wandb) 를 건너뛰므로 그 셀들이 하던 import 만 대신 해준다
    exec(compile("import os, sys, time", "<prelude>", "exec"), ns)
    for tag, src in cells:
        if tag not in BASE_CELLS:
            continue
        ctx = contextlib.redirect_stdout(io.StringIO()) if (quiet_base and tag != "8.") \
            else contextlib.nullcontext()
        with ctx:
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":                       # cfg 생성 직후 손잡이 적용
            for k, v in override.items():
                setattr(ns["cfg"], k, v)
            ns["cfg"].total_timesteps = steps
            ns["cfg"].eval_every = eval_every
            ns["cfg"].checkpoint_every = ckpt_every
            ns["cfg"].auto_resume = True      # 같은 run 디렉터리에서 끊기면 이어받는다
            print(f"[arm {arm} seed {seed}] override = {override}", flush=True)
        if tag == "4.":                       # 풀 생성 이후 시드 교체 → 풀은 고정
            ns["SEED"] = seed
            np.random.seed(seed % (2**31)); torch.manual_seed(seed)
    t0 = time.time()
    for tag, src in cells:
        if tag != TRAIN_CELL:
            continue
        exec(compile(src.replace("progress_bar=_progress", "progress_bar=False"),
                     "<nb:train>", "exec"), ns)
    res = dict(arm=arm, seed=seed, steps=steps, override=override,
               minutes=(time.time() - t0) / 60,
               pretrain=dict(ns.get("PRETRAIN_INFO", {})),
               history=ns["HISTORY"], best=ns["BEST"],
               final=ns["evaluate_routing"](ns["model"], ns["EVAL_POOL"]))
    res["final"]["h_mae_kg"] = ns["h_mae_kg"](ns["model"])
    if ns["cfg"].gnarl_mode:
        res["final"]["tau"] = float(ns["model"].policy.log_tau.detach().exp())
    json.dump(res, open("run_result.json", "w"), ensure_ascii=False, indent=1, default=float)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--ckpt-every", type=int, default=500_000)
    ap.add_argument("--unfreeze-at", type=int, default=None)
    ap.add_argument("--root", default=os.environ.get("STEP1_AB_ROOT", os.path.join(HERE, "..", "_ab")))
    ap.add_argument("--cache-dir", default=None, help="g3_scen_*.pkl 을 공유할 디렉터리")
    ap.add_argument("--child", action="store_true", help="내부용: 이 프로세스가 런 하나를 실행")
    ap.add_argument("--seed", type=int, help="내부용")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    cache_dir = os.path.abspath(a.cache_dir or os.path.join(root, "_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    extra = {}
    if a.unfreeze_at is not None:
        extra["gnarl_unfreeze_at"] = a.unfreeze_at

    if a.child:
        run_one(a.arm, a.seed, a.steps, a.eval_every, a.ckpt_every, extra, quiet_base=False)
        return

    tag = f"{a.arm}_{a.steps//1000}k"
    for seed in a.seeds:
        d = os.path.join(root, f"{tag}_s{seed}")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(cache_dir):        # 시나리오 캐시 공유 (생성 84초)
            if f.startswith("g3_scen_") and not os.path.exists(os.path.join(d, f)):
                os.symlink(os.path.join(cache_dir, f), os.path.join(d, f))
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--arm", a.arm,
               "--seed", str(seed), "--steps", str(a.steps),
               "--eval-every", str(a.eval_every), "--ckpt-every", str(a.ckpt_every)]
        if a.unfreeze_at is not None:
            cmd += ["--unfreeze-at", str(a.unfreeze_at)]
        print(f"\n===== {tag} seed={seed} → {d} =====", flush=True)
        with open(os.path.join(d, "run.log"), "w") as lg:
            p = subprocess.run(cmd, cwd=d, stdout=lg, stderr=subprocess.STDOUT)
        # 새로 만들어진 캐시는 공유 디렉터리로 올린다
        for f in os.listdir(d):
            if f.startswith("g3_scen_") and not os.path.islink(os.path.join(d, f)):
                os.replace(os.path.join(d, f), os.path.join(cache_dir, f))
                os.symlink(os.path.join(cache_dir, f), os.path.join(d, f))
        if p.returncode != 0:
            print(f"!! seed {seed} 실패 (rc={p.returncode}) — {d}/run.log 확인", flush=True)
            continue
        r = json.load(open(os.path.join(d, "run_result.json")))
        h = r["history"]
        print(f"   success: " + " ".join(f"{x['timesteps']//1000}k={x['success_rate']:.3f}" for x in h),
              flush=True)
        print(f"   tau:     " + " ".join(f"{x.get('tau', float('nan')):.4f}" for x in h), flush=True)


if __name__ == "__main__":
    main()
