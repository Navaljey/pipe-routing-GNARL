#!/usr/bin/env python3
"""g3_colab.ipynb 의 셀을 그대로 실행해 fail_cost 를 sweep 한다 (§19-2 재현용).

노트북이 유일한 소스다. 이 스크립트는 노트북 코드셀을 #@title 태그로 골라 exec 할 뿐,
환경/보상/평가 로직을 따로 갖고 있지 않다. 따라서 노트북을 고치면 이 결과도 같이 바뀐다.

사용:
    python3 run_fail_cost_sweep.py --seeds 20260919 777001 --steps 300000
    python3 run_fail_cost_sweep.py --modes none h_star          # 대조군
"""
import argparse, io, json, contextlib, os, sys, time, re

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g3_colab.ipynb")
# 학습 전까지 실행할 셀 (제목 앞 번호로 식별)
BASE_CELLS = ["2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "13.", "14."]
TRAIN_CELL = "15."

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

def run_one(cells, override, seed, steps, every):
    ns = {"__name__": "__main__"}
    for tag, src in cells:
        if tag not in BASE_CELLS:
            continue
        src = src.replace("USE_WANDB = True", "USE_WANDB = False")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "2.":
            for k, v in override.items():
                setattr(ns["cfg"], k, v)
    ns["SEED"] = seed                     # 시나리오 풀 생성 이후에 바꿔 풀은 고정
    ns["cfg"].total_timesteps, ns["cfg"].eval_every = steps, every
    t0 = time.time()
    for tag, src in cells:
        if tag != TRAIN_CELL:
            continue
        src = (src.replace("USE_WANDB = True", "USE_WANDB = False")
                  .replace("progress_bar=_progress", "progress_bar=False"))
        exec(compile(src, "<nb:train>", "exec"), ns)
    return dict(override=override, seed=seed, minutes=(time.time() - t0) / 60,
                history=ns["HISTORY"],
                final=ns["evaluate_routing"](ns["model"], ns["EVAL_POOL"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260919])
    ap.add_argument("--consts", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--modes", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--out", default="fail_cost_sweep_out.json")
    a = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch; torch.set_num_threads(1)

    cells = load_cells(NB)
    overrides = [dict(fail_cost_mode="const", fail_cost_const=c) for c in a.consts]
    overrides += [dict(fail_cost_mode=m) for m in a.modes]
    res = []
    for seed in a.seeds:
        for ov in overrides:
            print(f"=== {ov} seed={seed} ===", flush=True)
            r = run_one(cells, ov, seed, a.steps, a.eval_every)
            res.append(r)
            print("    " + json.dumps(r["final"], default=float), flush=True)
            json.dump(res, open(a.out, "w"), default=float, indent=1)
    print("DONE ->", a.out)

if __name__ == "__main__":
    main()
