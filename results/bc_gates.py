#!/usr/bin/env python3
"""§33-1 건전성 관문 ①③ — bc_batches=0 이면 수정 전 노트북과 셀 8 직후 상태가 같은가.

두 노트북(수정 전 = git HEAD~ 의 step1_train.ipynb, 수정 후 = 작업 트리)을 run_step1_ab.py 와 같은 방식으로
셀 3~8 까지 exec 하고 (H3fix 설정, 시드 1), policy 가중치 해시 · §11 체크리스트 출력을 비교한다.
사용: cd _ab/<빈 디렉터리(캐시 심볼릭 링크)> && python3 ../../results/bc_gates.py OLD.ipynb NEW.ipynb
"""
import contextlib, hashlib, io, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_step1_ab as R

OVR = dict(R.ARMS["L"], h_rank_batches=256, obs_align_norm="k", obs_capture=True, legacy_dj_edge_bug=False,
           bc_batches=0)   # 관문 ①: BC 를 끈 경로가 수정 전과 같은가 (기본값은 §33-6 에서 3000)


def state(nb_path, seed=1, extra=None):
    import numpy as np, torch
    ov = dict(OVR); ov.update(extra or {})
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec("import os, sys, time", ns)
    out = io.StringIO()
    for tag, src in R.load_cells(nb_path):
        if tag not in R.BASE_CELLS:
            continue
        with contextlib.redirect_stdout(out):
            exec(compile(src, f"<nb:{tag}>", "exec"), ns)
        if tag == "3.":
            for k, v in ov.items():
                setattr(ns["cfg"], k, v)
            ns["cfg"].total_timesteps = 0
        if tag == "4.":
            ns["SEED"] = seed; np.random.seed(seed); torch.manual_seed(seed)
    sd = ns["model"].policy.state_dict()
    hs = hashlib.sha256(b"".join(sd[k].detach().cpu().numpy().tobytes() for k in sorted(sd))).hexdigest()
    chk = [l for l in out.getvalue().splitlines() if "PASS" in l or "FAIL" in l]
    return hs, chk, ns


if __name__ == "__main__":
    old, new = sys.argv[1], sys.argv[2]
    h0, c0, _ = state(old)
    h1, c1, ns = state(new)
    res = dict(old_hash=h0, new_hash=h1, same_weights=h0 == h1, checklist_old=c0, checklist_new=c1,
               same_checklist=c0 == c1, n_pass=sum("PASS" in l for l in c1), n_fail=sum("FAIL" in l for l in c1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
