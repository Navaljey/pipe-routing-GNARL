#!/usr/bin/env python3
"""§21-4 순위 라벨을 병렬로 만들어 하나의 캐시(.npz)로 합친다.

노트북 셀 7b 의 `build_rank_dataset()` 을 그대로 쓴다 (노트북이 유일한 소스).
단일 프로세스로는 시나리오당 10~17초(총 약 83분)라 시나리오를 샤딩해 병렬로 돌린다.

사용: python3 build_rank_labels.py --workers 4 --out-dir <캐시 디렉터리>
"""
import argparse, contextlib, io, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402


def build_ns():
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in ("3.", "4.", "5.", "7.", "7b."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                ns["cfg"].pretrain_h = False; ns["cfg"].bc_batches = 0          # 라벨 생성에는 필요 없다
                ns["cfg"].h_rank_batches = 0          # 셀 7b 의 자동 빌드를 막는다
                ns["cfg"].h_rank_metrics = False
    return ns


def merge(parts):
    """샤드를 이어 붙인다. ptr 은 행 오프셋, pi/pj 는 전역 행 인덱스라 둘 다 보정한다."""
    obs = np.concatenate([p["obs"] for p in parts])
    dj = np.concatenate([p["dj"] for p in parts])
    t = np.concatenate([p["t"] for p in parts])
    ptr, pi, pj, off = [np.int64(0)], [], [], 0
    for p in parts:
        ptr.extend(p["ptr"][1:] + off)
        pi.append(p["pi"] + off); pj.append(p["pj"] + off)
        off += len(p["t"])
    return dict(obs=obs, dj=dj, t=t, ptr=np.asarray(ptr, dtype=np.int64),
                pi=np.concatenate(pi) if pi else np.zeros(0, np.int64),
                pj=np.concatenate(pj) if pj else np.zeros(0, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--shard", type=int, default=None, help="내부용")
    ap.add_argument("--nshards", type=int, default=None, help="내부용")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)

    ns = build_ns()
    cache = os.path.join(out_dir, ns["RANK_CACHE"])

    if a.shard is not None:                                  # ---- 워커 ----
        pool = ns["TRAIN_POOL"][a.shard::a.nshards]
        d = ns["build_rank_dataset"](pool, f"shard {a.shard}")
        np.savez(os.path.join(out_dir, f"_rank_shard{a.shard}.npz"), **d)
        return

    if os.path.exists(cache):
        print("이미 있다:", cache); return
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--shard", str(i),
                               "--nshards", str(a.workers), "--out-dir", out_dir],
                              stdout=open(os.path.join(out_dir, f"_rank_shard{i}.log"), "w"),
                              stderr=subprocess.STDOUT)
             for i in range(a.workers)]
    # 평가 풀(계기판 전용)은 본 프로세스가 맡는다 — 48개라 금방 끝난다
    e = ns["build_rank_dataset"](ns["EVAL_POOL"], "rank labels (eval)")
    for p in procs:
        assert p.wait() == 0, "샤드 실패 — _rank_shard*.log 확인"
    parts = [dict(np.load(os.path.join(out_dir, f"_rank_shard{i}.npz"))) for i in range(a.workers)]
    d = merge(parts)
    np.savez_compressed(cache, **d, **{("e_" + k): v for k, v in e.items()})
    for i in range(a.workers):
        os.remove(os.path.join(out_dir, f"_rank_shard{i}.npz"))
    print(f"{time.time()-t0:.0f}s -> {cache}")
    print(f"  학습: 결정시점 {len(d['ptr'])-1:,} · 이웃 {len(d['t']):,} · 쌍 {len(d['pi']):,}")
    print(f"  계기판: 결정시점 {len(e['ptr'])-1:,} · 이웃 {len(e['t']):,}")


if __name__ == "__main__":
    main()
