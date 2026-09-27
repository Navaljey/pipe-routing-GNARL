#!/usr/bin/env python3
"""§34-7 A-1 — "0.72 천장" 재측정 T1 · T2 (RL 없음). 조건은 CLAUDE.md §34-7 에 실행 전 고정.

T1  체크포인트의 h_core 를 §28-5 의 경로 밖 그룹(11,776)에 그대로 적용 (학습 없음)
T2  §31-7 지도 적합 틀에서 하나씩만 바꾼다: base(재현) · a CE 손실 · b 경로 위+밖 함께 · c C4 h 에서 시작

사용 (_ab/_cache 에서): python3 ../../results/a1_ceiling.py --out ../../results/step1_a1_ceiling.json
"""
import argparse, io, json, os, pickle, time, zipfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

CACHE = "step1_rank_100A_50x50x20_k6_n500+48_d2_f0.12_t180_m30_o80k_djfix_off.npz"
SCEN = "g3_scen_100A_50x50x20_k6_n500+48_d2_f0.12_t180_m30.pkl"
DIMS = np.array([49, 49, 19], float)
AB = ".."
TAU = 0.05
CKPTS = {"C4_300k": "L_300k_bc_s{}", "H3fix_300k": "L_300k_h3fix_s{}",
         "H3fix_2M": "L_2000k_h3fix_s{}", "C4_2M": "L_2000k_bc_s{}"}


def mk(i=80, wd=128, d=2):
    Ls = []
    for _ in range(d):
        Ls += [nn.Linear(i, wd), nn.ReLU()]; i = wd
    return nn.Sequential(*Ls, nn.Linear(i, 1))


def load_h(run):
    z = zipfile.ZipFile(os.path.join(AB, run, "checkpoint_step1_final.zip"))
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    net = mk(); net.load_state_dict({k[len("h_core."):]: v for k, v in sd.items() if k.startswith("h_core.")})
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--arms", default="base,a,b,c")
    a = ap.parse_args()
    torch.set_num_threads(4)
    train_pool, _ = pickle.load(open(SCEN, "rb"))
    goals = np.array([sc["goal"] for sc in train_pool])
    z = np.load(CACHE)
    obs, dj, t, ptr, nA = z["obs"].astype(np.float32), z["dj"], z["t"], z["ptr"], int(z["n_astar_rows"])
    gA = int(np.searchsorted(ptr, nA)); assert ptr[gA] == nA
    G_all = len(ptr) - 1
    cand = np.load("offpath_candidates.npz")["cand"]
    scen_off = np.concatenate([cand[w::4][:, 0] for w in range(4)])       # 경로 밖 그룹 순서 = 샤드 0..3 (§31-7)
    assert len(scen_off) == G_all - gA
    # 경로 밖 그룹 목표 = 시나리오 목표 검사
    g0 = np.rint(obs[ptr[gA:-1], 3:6] * DIMS).astype(int)
    assert np.array_equal(g0, goals[scen_off]), "경로 밖 그룹 시나리오 복원 실패"
    # 경로 위 그룹: 목표 좌표 → 후보 시나리오 목록
    gon = np.rint(obs[ptr[:gA], 3:6] * DIMS).astype(int)
    key = {}
    for i, g in enumerate(map(tuple, goals)):
        key.setdefault(g, []).append(i)
    on_cands = [key[tuple(g)] for g in gon]
    n_amb = sum(len(c) > 1 for c in on_cands)
    OFF = np.arange(gA, G_all); ON = np.arange(gA)
    X = torch.as_tensor(obs); DJ = torch.as_tensor(dj); T = t

    def top1_vec(net, groups):
        with torch.no_grad():
            h = F.softplus(net(X)).numpy()[:, 0]
        q = dj + h
        return np.array([np.argmin(q[ptr[g]:ptr[g + 1]]) == np.argmin(T[ptr[g]:ptr[g + 1]]) for g in groups], float)

    def split_sets(sp):
        test = set(np.random.default_rng(sp).choice(500, 100, replace=False).tolist())
        is_te = np.array([s in test for s in scen_off])
        return test, OFF[~is_te], OFF[is_te]

    out = {"T1": {}, "T2": {}, "meta": dict(n_off=int(len(OFF)), n_on=int(gA), n_on_goal_ambiguous=int(n_amb))}
    # ---------------- T1 ----------------
    _, tr0, te0 = split_sets(0)
    for name, pat in CKPTS.items():
        rows = []
        for s in (1, 2, 3, 4):
            net = load_h(pat.format(s))
            v = top1_vec(net, OFF)
            rows.append(dict(seed=s, all=float(v.mean()), heldout0=float(v[np.isin(OFF, te0)].mean()),
                             train0=float(v[np.isin(OFF, tr0)].mean())))
        out["T1"][name] = rows
        hm = np.array([r["heldout0"] for r in rows]); am = np.array([r["all"] for r in rows])
        print(f"T1 {name:11s} 전체 {am.mean():.4f} · 분할0 보류 {hm.mean():.4f} ± {hm.std(ddof=1)/2:.4f}", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)

    # ---------------- T2 ----------------
    def pairs(groups):
        pi, pj = [], []
        for g in groups:
            tt = T[ptr[g]:ptr[g + 1]]
            i, j = np.nonzero(tt[:, None] < tt[None, :] - 1e-9); pi.append(i + ptr[g]); pj.append(j + ptr[g])
        return np.concatenate(pi), np.concatenate(pj)

    Lmax = int(np.diff(ptr).max())
    IDX = np.full((G_all, Lmax), -1, np.int64); TGT = np.zeros((G_all, Lmax), np.float32)
    for g in range(G_all):
        a0, b0 = ptr[g], ptr[g + 1]
        IDX[g, :b0 - a0] = np.arange(a0, b0)
        tie = np.abs(T[a0:b0] - T[a0:b0].min()) < 1e-6
        TGT[g, :b0 - a0] = tie / tie.sum()
    IDXt, TGTt = torch.as_tensor(IDX), torch.as_tensor(TGT)

    arms = a.arms.split(",")
    for arm in arms:
        out["T2"][arm] = []
    for sp in range(a.splits):
        test, tr, te = split_sets(sp)
        for arm in arms:
            torch.manual_seed(1000 + sp)
            net = mk()
            if arm == "c":
                net = load_h(CKPTS["C4_300k"].format(sp % 4 + 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            gen = torch.Generator().manual_seed(2000 + sp)
            t0 = time.time(); extra = {}
            if arm == "a":
                trt = torch.as_tensor(tr)
                for _ in range(a.steps):
                    sel = trt[torch.randint(0, len(tr), (256,), generator=gen)]
                    ii = IDXt[sel]; valid = ii >= 0
                    h = F.softplus(net(X[ii[valid]]))[:, 0]
                    q = torch.full(ii.shape, float("inf")); q = q.masked_scatter(valid, DJ[ii[valid]] + h)
                    logp = torch.log_softmax(-q / TAU, dim=1).masked_fill(~valid, 0.0)
                    loss = -(TGTt[sel] * logp).sum(1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            else:
                groups = tr
                if arm == "b":
                    keep = [g for g in ON if all(c not in test for c in on_cands[g])]
                    extra = dict(n_on_train=len(keep),
                                 n_on_dropped_mixed=int(sum(1 for g in ON if any(c in test for c in on_cands[g])
                                                            and not all(c in test for c in on_cands[g]))))
                    groups = np.concatenate([np.asarray(keep, np.int64), tr])
                pi, pj = pairs(groups)
                for _ in range(a.steps):
                    k = torch.randint(0, len(pi), (2048,), generator=gen).numpy()
                    ia, ib = torch.as_tensor(pi[k]), torch.as_tensor(pj[k])
                    ha = F.softplus(net(X[ia]))[:, 0]; hb = F.softplus(net(X[ib]))[:, 0]
                    loss = F.softplus(((DJ[ia] + ha) - (DJ[ib] + hb)) / TAU).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            r = dict(split=sp, train=float(top1_vec(net, tr).mean()), heldout=float(top1_vec(net, te).mean()),
                     sec=round(time.time() - t0, 1), **extra)
            out["T2"][arm].append(r)
            print(f"T2 분할 {sp} {arm:>4}: 학습 {r['train']:.4f} · 보류 {r['heldout']:.4f} ({r['sec']:.0f}s) {extra}", flush=True)
            json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("→", a.out)


if __name__ == "__main__":
    main()
