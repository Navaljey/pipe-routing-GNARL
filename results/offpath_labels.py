#!/usr/bin/env python3
"""(b) 경로 밖 순위 라벨 (§28) — 정책이 실제로 방문한 A* 경로 밖 결정 시점에 A* 라벨을 붙인다.

§28-1 선정 규칙 그대로:
  출처 정책   (a) H3fix 300K 체크포인트 4개 (1회 수집, 오프라인)
  풀         학습 풀 500개만 (평가 풀은 쓰지 않는다)
  롤아웃      학습과 같은 확률적 행동 softmax(-f/τ), 시작 방향 sc["dirs"] 무작위, 체크포인트당 시나리오 1회
  대상        롤아웃 궤적 위 · 유효 이동 ≥ 2 · (pos,dir,s) 가 어떤 시작 방향 A* 최적 경로에도 없음
  중복        (시나리오, pos, dir, s) 1개로
  샘플링      시나리오당 최대 24개 균등 무작위 (고정 시드), 총 상한 12,000
  라벨        _rank_rows_for_state (참 ΔJ, A* 실패 이웃 있으면 버림)
  합치기      기존 A* 경로 위 라벨(_djfix) ∪ 경로 밖 라벨 → 쌍 합집합

사용 (_ab/_cache 에서):
  python3 offpath_labels.py collect ../L_300k_h3fix_s1 ../L_300k_h3fix_s2 ../L_300k_h3fix_s3 ../L_300k_h3fix_s4
  python3 offpath_labels.py label --workers 4
출력: offpath_candidates.npz → <RANK_CACHE 이름>_off.npz (노트북 cfg.h_rank_offpath=True 가 읽는다)
"""
import argparse, contextlib, io, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

PER_SCEN, TOTAL, SEED = 24, 12_000, 20260924
CAND = "offpath_candidates.npz"


def boot(upto=("3.", "4.", "5.", "6.", "7b.")):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time, torch\nimport torch.nn as nn\nimport torch.nn.functional as F", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in upto:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                c = ns["cfg"]
                c.pretrain_h = False; c.bc_batches = 0; c.h_rank_batches = 0; c.h_rank_metrics = False
                c.legacy_dj_edge_bug = False; c.obs_capture = True; c.obs_align_norm = "k"; c.h_rank_offpath = False
    return ns


def collect(dirs):
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    ns = boot(("3.", "4.", "5.", "6.", "7.", "7b.", "8."))  # 7·8: 정책 클래스 정의 (MaskablePPO.load 용)
    pool = ns["TRAIN_POOL"]
    on_path = [set((tuple(int(v) for v in p), int(d), int(s)) for sol in sc["astar"].values()
                   for (p, d, s, _) in sol["states"]) for sc in pool]
    seen = {}                                              # (scen, pos, dir, s) -> g_J (첫 방문)
    n_ep = n_dec = n_off = 0
    t0 = time.time()
    for di, d in enumerate(dirs):
        model = MaskablePPO.load(os.path.join(d, "checkpoint_step1_final.zip"), device="cpu", print_system_info=False)
        torch.manual_seed(SEED + di)
        env = ns["PipeRoutingEnvStep1"](pool, eval_mode=False, seed=SEED + di)
        for i in range(len(pool)):
            obs, _ = env.reset(options={"index": i})
            done = False
            while not done:
                m = env.action_masks()
                st = (tuple(int(v) for v in env.pos), int(env.dir), int(env.s))
                if m[:26].sum() >= 2:
                    n_dec += 1
                    if st not in on_path[i]:
                        n_off += 1
                        seen.setdefault((i,) + st, float(env.g_J))
                with torch.no_grad():
                    dist = model.policy.get_distribution(torch.as_tensor(obs[None]), action_masks=m[None])
                    a = int(dist.get_actions(deterministic=False)[0])      # 학습과 같은 확률적 행동
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            n_ep += 1
        print(f"  {os.path.basename(d.rstrip('/'))}: 누적 에피소드 {n_ep} · 결정 {n_dec:,} · 경로 밖 {n_off:,} · "
              f"고유 경로 밖 {len(seen):,} ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)
    by_scen = {}
    for key in seen:
        by_scen.setdefault(key[0], []).append(key)
    picked = []
    for i in sorted(by_scen):
        ks = sorted(by_scen[i])
        sel = rng.choice(len(ks), size=min(PER_SCEN, len(ks)), replace=False)
        picked += [ks[j] for j in sorted(sel)]
    if len(picked) > TOTAL:
        picked = [picked[j] for j in sorted(rng.choice(len(picked), TOTAL, replace=False))]
    arr = np.array([[k[0], *k[1], k[2], k[3]] for k in picked], dtype=np.int64)   # scen, x, y, z, dir, s
    g = np.array([seen[k] for k in picked], dtype=np.float64)
    np.savez(CAND, cand=arr, g=g)
    stats = dict(episodes=n_ep, decisions=n_dec, offpath_decisions=n_off, offpath_share=n_off / max(n_dec, 1),
                 unique_offpath=len(seen), scenarios_with_offpath=len(by_scen), picked=len(picked))
    json.dump(stats, open("offpath_collect_stats.json", "w"), indent=1)
    print("수집:", stats, "→", CAND)


def label_shard(shard, nshards):
    ns = boot()
    z = np.load(CAND); cand, g = z["cand"][shard::nshards], z["g"][shard::nshards]
    pool = ns["TRAIN_POOL"]
    obs, dj, t, ptr, dropped = [], [], [], [0], 0
    for (i, x, y, zz, d, s), gg in zip(cand, g):
        rows = ns["_rank_rows_for_state"](pool[int(i)], (int(x), int(y), int(zz)), int(d), int(s), float(gg))
        if not rows:
            dropped += 1; continue
        for (o, a, b) in rows:
            obs.append(o); dj.append(a); t.append(b)
        ptr.append(len(obs))
    np.savez(f"_off_shard{shard}.npz", obs=np.stack(obs).astype(np.float32), dj=np.float32(dj), t=np.float32(t),
             ptr=np.int64(ptr), dropped=np.int64(dropped))


def label(workers):
    ns = boot()
    base_path = ns["RANK_CACHE"]
    assert base_path.endswith("_djfix.npz") and os.path.exists(base_path), f"기준 캐시 없음: {base_path}"
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "_shard", str(w), str(workers)],
                              stdout=open(f"_off_shard{w}.log", "w"), stderr=subprocess.STDOUT) for w in range(workers)]
    for p in procs:
        assert p.wait() == 0, "샤드 실패 — _off_shard*.log 확인"
    base = dict(np.load(base_path))
    obs, dj, t, ptr = [base["obs"]], [base["dj"]], [base["t"]], [base["ptr"]]
    off = len(base["t"]); n_groups, dropped = 0, 0
    for w in range(workers):
        p = np.load(f"_off_shard{w}.npz")
        obs.append(p["obs"]); dj.append(p["dj"]); t.append(p["t"]); ptr.append(p["ptr"][1:] + off)
        off += len(p["t"]); n_groups += len(p["ptr"]) - 1; dropped += int(p["dropped"])
    out = dict(base)
    out["obs"], out["dj"], out["t"], out["ptr"] = np.concatenate(obs), np.concatenate(dj), np.concatenate(t), np.concatenate(ptr)
    n_base_rows = len(base["t"])
    pi, pj = [base["pi"]], [base["pj"]]
    P = out["ptr"]; T = out["t"]
    start_group = len(base["ptr"]) - 1
    for s0, s1 in zip(P[start_group:-1], P[start_group + 1:]):
        tt = T[s0:s1]
        ii, jj = np.nonzero(tt[:, None] < tt[None, :] - 1e-9)
        pi.append(ii + s0); pj.append(jj + s0)
    out["pi"], out["pj"] = np.concatenate(pi).astype(np.int64), np.concatenate(pj).astype(np.int64)
    out["n_astar_rows"] = np.int64(n_base_rows)             # 앞쪽 = A* 경로 위, 뒤쪽 = 경로 밖
    dst = base_path.replace(".npz", "_off.npz")
    np.savez_compressed(dst, **out)
    for w in range(workers):
        os.remove(f"_off_shard{w}.npz")
    print(f"라벨: 경로 밖 결정 시점 {n_groups:,} (A* 실패로 버림 {dropped}) · 이웃 {len(out['t']) - n_base_rows:,} · "
          f"쌍 {len(base['pi']):,} → {len(out['pi']):,} ({time.time() - t0:.0f}s) → {dst}")
    st = json.load(open("offpath_collect_stats.json"))
    st.update(label_groups=n_groups, label_dropped=dropped, label_rows=int(len(out["t"]) - n_base_rows),
              pairs_astar=int(len(base["pi"])), pairs_total=int(len(out["pi"])))
    json.dump(st, open("offpath_collect_stats.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "_shard":
        label_shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["collect", "label"])
        ap.add_argument("dirs", nargs="*")
        ap.add_argument("--workers", type=int, default=4)
        a = ap.parse_args()
        collect(a.dirs) if a.cmd == "collect" else label(a.workers)
