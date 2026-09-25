#!/usr/bin/env python3
"""§31-7 H2 사전 점검 — R-OBS 를 열 가치가 있는지 재는 측정 (관측 채널 제안이 아니다).

§28-5(3) 의 오프라인 순수 지도 적합을 H2 후보 채널(H2-a/b/c)을 붙인 관측으로 다시 돌려,
경로 밖 **보류 시나리오 top-1** 이 80차원 기준(0.721)보다 오르는지 본다. RL 없음 · 노트북/정책 무변경.
후보 채널은 이 스크립트 안에서만 계산한다 (§31-7 의 고정 정의 그대로).

  H2-a  26방향 연속 자유 칸 len(1~24 검사) → clip((len−7)/(24−7), 0, 1)                    +26
  H2-b  26방향 × 고리 c∈{8,20}: 중심 pos+c·d, 한 변 8칸 정육면체의 막힌 칸 비율              +52
  H2-c  목표에서의 26-연결 octile 최단거리 geo (장애물만) → tanh(geo/24), tanh((geo−octile)/7) +2
격자 밖 = 막힘. 분할: 시나리오 단위 80/20 × 5 (분할 시드 0~4, 0 = §28-5). 분할마다 기준과 세 인코딩을 같은 학습 시드로 짝짓는다.

사용 (_ab/_cache 에서): python3 h2_precheck.py [--splits 5] [--steps 6000] --out ../../results/step1_h2_precheck.json
"""
import argparse, contextlib, io, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

CACHE = "step1_rank_100A_50x50x20_k6_n500+48_d2_f0.12_t180_m30_o80k_djfix_off.npz"
L_RAY, NEAR, BOX, RINGS, L_GEO, K1 = 24, 7, 8, (8, 20), 24.0, 7.0


def boot():
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time, torch\nimport torch.nn as nn\nimport torch.nn.functional as F", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in ("3.", "4."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                c = ns["cfg"]
                c.pretrain_h = False; c.h_rank_batches = 0; c.h_rank_metrics = False
                c.legacy_dj_edge_bug = False; c.obs_capture = True; c.obs_align_norm = "k"
    return ns


def geodesic(free, goal, dirs, w):
    """목표에서의 26-연결 최단거리 (octile 가중, 장애물·격자 밖 통과 불가). numpy 완화 반복 — 수렴까지."""
    P = 1
    fp = np.zeros(tuple(s + 2 * P for s in free.shape), bool); fp[P:-P, P:-P, P:-P] = free
    D = np.full(fp.shape, np.inf); D[tuple(np.asarray(goal) + P)] = 0.0
    X, Y, Z = free.shape
    core = (slice(P, P + X), slice(P, P + Y), slice(P, P + Z))
    it = 0
    while True:
        it += 1
        cur = D[core]; best = cur.copy()
        for (dx, dy, dz), ww in zip(dirs, w):
            nb = D[P + dx:P + dx + X, P + dy:P + dy + Y, P + dz:P + dz + Z] + ww
            np.minimum(best, nb, out=best)
        best[~free] = np.inf
        if np.array_equal(best, cur):
            return D[core], it
        D[core] = best


def octile(p, g):
    a = np.sort(np.abs(p - g), axis=1)[:, ::-1]
    return (a[:, 0] - a[:, 1]) + 2 ** .5 * (a[:, 1] - a[:, 2]) + 3 ** .5 * a[:, 2]


def channels(pool, rows_scen, pos, dirs, w):
    """행(이웃 상태)마다 H2-a/b/c 채널. pos (N,3) int, rows_scen (N,) 시나리오 번호."""
    N = len(pos)
    A = np.zeros((N, 26), np.float32); B = np.zeros((N, 52), np.float32); C = np.zeros((N, 2), np.float32)
    D = np.asarray(dirs, np.int64)
    Pp = max(L_RAY, max(RINGS) + BOX) + 1
    t_geo, iters = [], []
    for s in np.unique(rows_scen):
        idx = np.where(rows_scen == s)[0]; sc = pool[int(s)]
        free = np.asarray(sc["free"], bool); goal = np.asarray(sc["goal"], np.int64)
        fp = np.zeros(tuple(x + 2 * Pp for x in free.shape), bool); fp[Pp:-Pp, Pp:-Pp, Pp:-Pp] = free
        p = pos[idx] + Pp
        # H2-a: 연속 자유 칸 수 (1..L_RAY)
        steps = np.arange(1, L_RAY + 1)
        cells = p[:, None, None, :] + D[None, :, None, :] * steps[None, None, :, None]      # (n,26,L,3)
        fr = fp[cells[..., 0], cells[..., 1], cells[..., 2]]
        ln = np.cumprod(fr, axis=2).sum(axis=2)
        A[idx] = np.clip((ln - NEAR) / (L_RAY - NEAR), 0, 1)
        # H2-b: 3D 누적합으로 정육면체 막힌 칸 수
        blk = (~fp).astype(np.int32)
        S = np.zeros(tuple(x + 1 for x in blk.shape), np.int32); S[1:, 1:, 1:] = blk.cumsum(0).cumsum(1).cumsum(2)
        cols = []
        for c in RINGS:
            lo = p[:, None, :] + D[None] * c - BOX // 2; hi = lo + BOX                   # [lo, hi)
            x0, y0, z0 = lo[..., 0], lo[..., 1], lo[..., 2]; x1, y1, z1 = hi[..., 0], hi[..., 1], hi[..., 2]
            cnt = (S[x1, y1, z1] - S[x0, y1, z1] - S[x1, y0, z1] - S[x1, y1, z0]
                   + S[x0, y0, z1] + S[x0, y1, z0] + S[x1, y0, z0] - S[x0, y0, z0])
            cols.append(cnt / BOX ** 3)
        B[idx] = np.concatenate(cols, axis=1)
        # H2-c: 지오데식
        t0 = time.time(); G, it = geodesic(free, goal, dirs, w); t_geo.append(time.time() - t0); iters.append(it)
        g = G[pos[idx, 0], pos[idx, 1], pos[idx, 2]]
        o = octile(pos[idx].astype(float), goal.astype(float))
        C[idx, 0] = np.where(np.isfinite(g), np.tanh(g / L_GEO), 1.0)
        C[idx, 1] = np.where(np.isfinite(g), np.tanh(np.maximum(g - o, 0) / K1), 1.0)
    return A, B, C, dict(geo_sec_mean=float(np.mean(t_geo)), geo_sec_max=float(np.max(t_geo)),
                         geo_iters_mean=float(np.mean(iters)), n_scen=int(len(t_geo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--arms", default="base,a,b,c")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(4)
    ns = boot()
    pool, DIMS, DIRS = ns["TRAIN_POOL"], np.asarray(ns["DIMS"]), [tuple(int(v) for v in d) for d in ns["DIRS_ARR"]]
    w = [float(np.sqrt(sum(v * v for v in d))) for d in DIRS]
    z = np.load(CACHE); obs, dj, t, ptr, nA = z["obs"], z["dj"], z["t"], z["ptr"], int(z["n_astar_rows"])
    gA = int(np.searchsorted(ptr, nA)); assert ptr[gA] == nA
    cand = np.load("offpath_candidates.npz")["cand"]
    scen_g = np.concatenate([cand[w_::4][:, 0] for w_ in range(4)])                    # 경로 밖 그룹 순서 = 샤드 0..3
    G = len(ptr) - 1 - gA; assert len(scen_g) == G
    r0 = nA
    O = obs[r0:].astype(np.float32); DJ = dj[r0:]; T = t[r0:]; P = ptr[gA:] - r0
    rows_scen = np.repeat(scen_g, np.diff(P))
    pos = np.rint(O[:, 0:3] * DIMS).astype(np.int64)
    # 위치 복원 검사: 그룹의 부모 상태(후보) + 자식 방향 = 행 위치
    cand_order = np.concatenate([cand[w_::4] for w_ in range(4)])
    child_dir = O[:, 6:32].argmax(1)
    parent = np.repeat(cand_order[:, 1:4], np.diff(P), axis=0)
    assert np.array_equal(parent + np.asarray(DIRS)[child_dir], pos), "행 위치 복원 실패"
    goal_rows = np.rint(O[:, 3:6] * DIMS).astype(np.int64)
    assert np.array_equal(goal_rows, np.stack([np.asarray(pool[int(s)]["goal"]) for s in rows_scen])), "목표 불일치"
    t0 = time.time()
    A, B, C, geo_stat = channels(pool, rows_scen, pos, DIRS, w)
    print(f"채널 계산 {time.time() - t0:.0f}s · 지오데식 시나리오당 {geo_stat['geo_sec_mean']*1000:.0f} ms "
          f"(최대 {geo_stat['geo_sec_max']*1000:.0f}) · 반복 {geo_stat['geo_iters_mean']:.0f}", flush=True)
    feats = {"base": O, "a": np.concatenate([O, A], 1), "b": np.concatenate([O, B], 1), "c": np.concatenate([O, C], 1)}
    arms = a.arms.split(",")

    def pairs(groups):
        pi, pj = [], []
        for g in groups:
            s0, s1 = P[g], P[g + 1]; tt = T[s0:s1]
            i, j = np.nonzero(tt[:, None] < tt[None, :] - 1e-9); pi.append(i + s0); pj.append(j + s0)
        return np.concatenate(pi), np.concatenate(pj)

    def top1(net, X, groups):
        with torch.no_grad():
            h = F.softplus(net(X)).numpy()[:, 0]
        q = DJ + h
        return float(np.mean([np.argmin(q[P[g]:P[g + 1]]) == np.argmin(T[P[g]:P[g + 1]]) for g in groups]))

    def mk(i, wd=128, d=2):
        Ls = []
        for _ in range(d):
            Ls += [nn.Linear(i, wd), nn.ReLU()]; i = wd
        return nn.Sequential(*Ls, nn.Linear(i, 1))

    res = {arm: [] for arm in arms}
    DJt = torch.as_tensor(DJ)
    for sp in range(a.splits):
        test_s = set(np.random.default_rng(sp).choice(500, 100, replace=False).tolist())
        te = np.array([g for g in range(G) if scen_g[g] in test_s]); tr = np.array([g for g in range(G) if scen_g[g] not in test_s])
        pi, pj = pairs(tr)
        for arm in arms:
            X = torch.as_tensor(feats[arm])
            torch.manual_seed(1000 + sp)
            net = mk(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            g = torch.Generator().manual_seed(2000 + sp)
            t0 = time.time()
            for _ in range(a.steps):
                k = torch.randint(0, len(pi), (2048,), generator=g).numpy()
                ia, ib = torch.as_tensor(pi[k]), torch.as_tensor(pj[k])
                ha = F.softplus(net(X[ia]))[:, 0]; hb = F.softplus(net(X[ib]))[:, 0]
                loss = F.softplus(((DJt[ia] + ha) - (DJt[ib] + hb)) / 0.05).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            r = dict(split=sp, train=top1(net, X, tr), heldout=top1(net, X, te), n_tr=int(len(tr)), n_te=int(len(te)),
                     sec=round(time.time() - t0, 1))
            res[arm].append(r)
            print(f"분할 {sp} {arm:>4}: 학습 {r['train']:.3f} · 보류 {r['heldout']:.3f} ({r['sec']:.0f}s)", flush=True)
        json.dump(dict(results=res, geo=geo_stat, dims={k_: int(v.shape[1]) for k_, v in feats.items()}),
                  open(a.out, "w"), ensure_ascii=False, indent=1)
    print("→", a.out)


if __name__ == "__main__":
    main()
