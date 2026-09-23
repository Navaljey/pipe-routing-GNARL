#!/usr/bin/env python3
"""끝난 런 디렉터리에서 노트북 셀 10(§17 대시보드 + 졸업 판정)만 다시 그린다.

학습은 하지 않는다. 셀 3~8 로 환경/모델을 세우고 (auto_resume 이 checkpoint_step1_final.zip
을 집는다), 셀 9 의 정의부만 실행한 뒤 (학습 호출 직전에서 자른다) step1_history.json 의
평가 이력을 HISTORY 에 넣고 셀 10 을 돌린다.

사용: cd <런 디렉터리> && python3 <repo>/results/make_dashboard.py [--arm A --set learning_rate=1e-4]
"""
import argparse, contextlib, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB, ARMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="A")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--fresh-eval", action="store_true",
                help="체크포인트를 현재 코드로 다시 평가해 이력 끝에 붙인다 (지표 정의가 바뀐 뒤 재판정용)")
a = ap.parse_args()

override = dict(ARMS.get(a.arm, {}))
for kv in a.set:
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except Exception:
        pass
    override[k] = v

ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
exec(compile("import os, sys, time", "<prelude>", "exec"), ns)
for tag, src in load_cells(NB):
    if tag in ("3.", "4.", "5.", "6.", "7.", "7b.", "8."):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":
            for k, v in override.items():
                setattr(ns["cfg"], k, v)
            ns["cfg"].pretrain_h = False          # 체크포인트의 h 를 그대로 본다
    elif tag == "9.":
        head = src.split("_t0 = time.time()")[0]  # 학습 호출 직전까지만
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(head, "<nb:9-defs>", "exec"), ns)
    elif tag == "10.":
        hist = json.load(open("step1_history.json"))
        ns["HISTORY"][:] = hist["history"]
        ns["BEST"].update(hist["best"])
        if a.fresh_eval:
            m = ns["evaluate_routing"](ns["model"], ns["EVAL_POOL"])
            m["timesteps"] = int(ns["model"].num_timesteps)
            m["h_mae_kg"] = ns["h_mae_kg"](ns["model"])
            m.update(ns["h_rank_metrics"](ns["model"].policy))
            ns["HISTORY"].append(m)
            print(f"현재 코드로 재평가 → 이력 끝에 추가 ({m['timesteps']:,} 스텝)")
        print(f"평가 이력 {len(ns['HISTORY'])}개 로드 "
              f"({ns['HISTORY'][0]['timesteps']:,} ~ {ns['HISTORY'][-1]['timesteps']:,} 스텝)\n")
        exec(compile(src, "<nb:10>", "exec"), ns)
        break
