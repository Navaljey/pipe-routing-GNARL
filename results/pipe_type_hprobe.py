#!/usr/bin/env python3
"""§38-1 ① — h_net 이 관종별 남은 J 를 추정하는가 (오프라인 지도 적합 · RL 없음).

같은 결정 시점을 압력관·중력관(G0) 두 관종으로 라벨링하고, 관종 플래그(obs[80]) 가 있는 입력 / 지운 입력으로
h_core 크기의 MLP 를 순위 손실로 적합해 **분기 집합 D**(관종에 따라 최선 행동이 갈리는 시점)의 보류 top-1 을 비교한다.

  label: python3 ../../results/pipe_type_hprobe.py label --out ../../results/_pt/hprobe_labels.npz
  fit  : python3 ../../results/pipe_type_hprobe.py fit --labels ../../results/_pt/hprobe_labels.npz \
                 --out ../../results/step1_pipe_type_hprobe.json
(_ab/_cache 에서 실행 — Step 1 시나리오 캐시를 쓴다. 노트북은 바꾸지 않는다.)
"""
import argparse, contextlib, io, json, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

NS = None
Y_UNREACH = 3.0     # 정규화 J — 도달 불가 이웃은 순위 끝 (§38-1)


def boot():
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in ("3.", "4."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                c = ns["cfg"]
                c.pretrain_h = False; c.bc_batches = 0; c.h_rank_batches = 0; c.h_rank_metrics = False
                c.obs_pipe_type = True; c.gravity_frac = 0.0      # Step 1 풀 그대로 · 82차원 관측
    assert ns["OBS_DIM"] == 82, ns["OBS_DIM"]
    return ns


def label_one(idx):
    ns = NS
    sc = ns["TRAIN_POOL"][idx]
    K, DIRS26, J_MAX = ns["K"], ns["DIRS26"], ns["J_MAX"]
    mask, mJ_all, astar, bobs = ns["action_mask_27"], ns["move_J_all"], ns["astar_gnarl"], ns["build_obs"]
    pad, goal, start = sc["pad"], tuple(sc["goal"]), tuple(sc["start"])
    states = {}
    for d, sol in sc["astar"].items():
        for (p, dd, ss, gg) in sol["states"]:
            states.setdefault((tuple(p), int(dd), int(ss)), float(gg))
    n_p = len(states); g_fail = 0
    for d in sc["astar"].keys():
        r = astar(pad, start, goal, int(d), gravity=True)
        if r is None:
            g_fail += 1; continue
        for (p, dd, ss, gg) in r["states"]:
            states.setdefault((tuple(p), int(dd), int(ss)), float(gg))
    memo = {}; n_unreach = 0; n_astar = 0
    obs, dj, t, act, meta = [], [], [], [], []      # meta: (시점 번호, 관종, 행 시작, 행 끝)
    for sid, ((pos, dd, ss), gg) in enumerate(states.items()):
        if pos == goal:
            continue
        dJ_all = mJ_all(dd)
        for T in (0, 1):
            m = mask(pad, pos, dd, ss, goal, K, gravity=bool(T))
            cand = [int(i) for i in np.where(m[:26])[0]]
            if len(cand) < 2:
                continue
            r0 = len(t)
            for i in cand:
                dv = DIRS26[i]
                npos = (pos[0] + dv[0], pos[1] + dv[1], pos[2] + dv[2])
                nss = min(ss + 1, K) if i == dd else 1
                a = float(dJ_all[i]) / J_MAX
                if npos == goal:
                    y = 0.0
                else:
                    key = (T, npos, i, nss)
                    if key not in memo and T == 1 and npos[2] < goal[2]:
                        memo[key] = Y_UNREACH; n_unreach += 1      # G0 는 오를 수 없다 — 정확한 판정
                    if key not in memo:
                        rr = astar(pad, npos, goal, i, s0=nss, gravity=bool(T)); n_astar += 1
                        memo[key] = Y_UNREACH if rr is None else float(rr["J"]) / J_MAX
                        n_unreach += rr is None
                    y = memo[key]
                obs.append(bobs(pad, npos, i, nss, gg + float(dJ_all[i]), goal, gravity=bool(T)))
                dj.append(a); t.append(a + y); act.append(i)
            meta.append((sid, T, r0, len(t)))
    return dict(idx=idx, obs=np.asarray(obs, np.float32).reshape(-1, 82), dj=np.asarray(dj, np.float32),
                t=np.asarray(t, np.float32), act=np.asarray(act, np.int16), meta=np.asarray(meta, np.int64).reshape(-1, 4),
                n_states=len(states), n_pressure_states=n_p, g_fail=g_fail, n_unreach=n_unreach, n_astar=n_astar)


def cmd_label(a):
    global NS
    NS = boot()
    pool = NS["TRAIN_POOL"]
    idxs = [i for i, sc in enumerate(pool) if sc["goal"][2] <= sc["start"][2]]
    print(f"시나리오 {len(idxs)} / {len(pool)} (목표가 시작보다 높지 않음)", flush=True)
    if a.limit:
        idxs = idxs[:a.limit]
    t0 = time.time(); res = []
    with Pool(a.procs) as p:
        for k, r in enumerate(p.imap_unordered(label_one, idxs, chunksize=1)):
            res.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k+1}/{len(idxs)}  {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["idx"])
    obs, dj, t, act, meta, scen = [], [], [], [], [], []
    off = 0; soff = 0
    for r in res:
        m = r["meta"].copy(); m[:, 2:] += off; m[:, 0] += soff
        obs.append(r["obs"]); dj.append(r["dj"]); t.append(r["t"]); act.append(r["act"]); meta.append(m)
        scen.append(np.full(len(m), r["idx"], np.int64))
        off += len(r["t"]); soff += r["n_states"]
    stats = dict(n_scen=len(res), n_states=int(sum(r["n_states"] for r in res)),
                 n_pressure_states=int(sum(r["n_pressure_states"] for r in res)),
                 g_astar_fail=int(sum(r["g_fail"] for r in res)), n_unreach=int(sum(r["n_unreach"] for r in res)),
                 n_astar=int(sum(r["n_astar"] for r in res)), sec=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, obs=np.concatenate(obs), dj=np.concatenate(dj), t=np.concatenate(t),
                        act=np.concatenate(act), meta=np.concatenate(meta), scen=np.concatenate(scen),
                        stats=json.dumps(stats))
    print("done", stats, flush=True)


def cmd_fit(a):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(4)
    z = np.load(a.labels)
    O, DJ, T, ACT, META, SCEN = z["obs"], z["dj"], z["t"], z["act"], z["meta"], z["scen"]
    stats = json.loads(str(z["stats"]))
    G = len(META)
    sid, typ, s0, s1 = META[:, 0], META[:, 1], META[:, 2], META[:, 3]
    real_y_max = float((T - DJ)[(T - DJ) < Y_UNREACH - 1e-6].max())
    # 분기 집합 D: 두 관종 그룹이 모두 있는 시점 중, 중력관 유효 이웃 안에서 argmin t_p ≠ argmin t_g
    by = {}
    for g in range(G):
        by.setdefault(int(sid[g]), {})[int(typ[g])] = g
    D_groups, D_states, both = set(), 0, 0
    for s, d in by.items():
        if 0 not in d or 1 not in d:
            continue
        both += 1
        gp, gg = d[0], d[1]
        ap, tp = ACT[s0[gp]:s1[gp]], T[s0[gp]:s1[gp]]
        ag, tg = ACT[s0[gg]:s1[gg]], T[s0[gg]:s1[gg]]
        tp_on_g = np.array([tp[np.where(ap == x)[0][0]] for x in ag])
        if ag[np.argmin(tp_on_g)] != ag[np.argmin(tg)]:
            D_states += 1; D_groups.update([gp, gg])
    isD = np.array([g in D_groups for g in range(G)])
    print(f"그룹 {G:,} (압력 {int((typ==0).sum()):,} · 중력 {int((typ==1).sum()):,}) · 두 관종 시점 {both:,} · "
          f"D {D_states:,} ({D_states/max(both,1):.3f}) · 실제 y 최대 {real_y_max:.3f}", flush=True)

    def pairs(groups):
        pi, pj = [], []
        for g in groups:
            tt = T[s0[g]:s1[g]]
            i, j = np.nonzero(tt[:, None] < tt[None, :] - 1e-9); pi.append(i + s0[g]); pj.append(j + s0[g])
        return np.concatenate(pi), np.concatenate(pj)

    def top1s(net, X, groups):
        with torch.no_grad():
            h = F.softplus(net(X)).numpy()[:, 0]
        q = DJ + h
        return np.array([np.argmin(q[s0[g]:s1[g]]) == np.argmin(T[s0[g]:s1[g]]) for g in groups], float)

    def mk(i, wd=128, d=2):
        Ls = []
        for _ in range(d):
            Ls += [nn.Linear(i, wd), nn.ReLU()]; i = wd
        return nn.Sequential(*Ls, nn.Linear(i, 1))

    Xf = torch.as_tensor(O)
    O0 = O.copy(); O0[:, 80] = 0.0; Xn = torch.as_tensor(O0)
    DJt = torch.as_tensor(DJ)
    uscen = np.unique(SCEN)
    res = {"flag": [], "noflag": []}
    for sp in range(a.splits):
        test_s = set(np.random.default_rng(sp).choice(uscen, len(uscen) // 5, replace=False).tolist())
        te = np.array([g for g in range(G) if SCEN[g] in test_s]); tr = np.array([g for g in range(G) if SCEN[g] not in test_s])
        pi, pj = pairs(tr)
        for arm, X in (("flag", Xf), ("noflag", Xn)):
            torch.manual_seed(1000 + sp)
            net = mk(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            gen = torch.Generator().manual_seed(2000 + sp)
            t0 = time.time()
            for _ in range(a.steps):
                k = torch.randint(0, len(pi), (2048,), generator=gen).numpy()
                ia, ib = torch.as_tensor(pi[k]), torch.as_tensor(pj[k])
                ha = F.softplus(net(X[ia]))[:, 0]; hb = F.softplus(net(X[ib]))[:, 0]
                loss = F.softplus(((DJt[ia] + ha) - (DJt[ib] + hb)) / 0.05).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            tt_te, tt_tr = top1s(net, X, te), top1s(net, X, tr)
            dte, dtr = isD[te], isD[tr]; pte = typ[te]
            r = dict(split=sp, sec=round(time.time() - t0, 1), n_te=int(len(te)), n_te_D=int(dte.sum()),
                     heldout_D=float(tt_te[dte].mean()), heldout_D_pressure=float(tt_te[dte & (pte == 0)].mean()),
                     heldout_D_gravity=float(tt_te[dte & (pte == 1)].mean()),
                     heldout_all=float(tt_te.mean()), heldout_pressure=float(tt_te[pte == 0].mean()),
                     heldout_gravity=float(tt_te[pte == 1].mean()),
                     train_D=float(tt_tr[dtr].mean()), train_all=float(tt_tr.mean()))
            res[arm].append(r)
            print(f"분할 {sp} {arm:>6}: 보류 D {r['heldout_D']:.3f} (압력 {r['heldout_D_pressure']:.3f} · 중력 {r['heldout_D_gravity']:.3f}) "
                  f"· 보류 전체 {r['heldout_all']:.3f} · 학습 D {r['train_D']:.3f} ({r['sec']:.0f}s)", flush=True)
    d = np.array([f["heldout_D"] - n["heldout_D"] for f, n in zip(res["flag"], res["noflag"])])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    verdict = "통과" if d.mean() > 2 * se else "미통과"
    summ = dict(delta_D_mean=float(d.mean()), delta_D_se=se, delta_by_split=d.tolist(), verdict=verdict,
                n_groups=G, n_both=both, n_D=D_states, D_frac=D_states / max(both, 1), real_y_max=real_y_max,
                label_stats=stats)
    for arm in res:
        for k in ("heldout_D", "heldout_D_pressure", "heldout_D_gravity", "heldout_all", "heldout_pressure",
                  "heldout_gravity", "train_D", "train_all"):
            v = np.array([r[k] for r in res[arm]]); summ[f"{arm}_{k}"] = [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]
    json.dump(dict(summary=summ, results=res), open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nΔ(D, 보류) = {d.mean():+.4f} ± {se:.4f} → {verdict}")
    print("→", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label"); l.add_argument("--out", required=True); l.add_argument("--procs", type=int, default=4)
    l.add_argument("--limit", type=int, default=0)
    f = sub.add_parser("fit"); f.add_argument("--labels", required=True); f.add_argument("--out", required=True)
    f.add_argument("--splits", type=int, default=5); f.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    cmd_label(a) if a.cmd == "label" else cmd_fit(a)


if __name__ == "__main__":
    main()
