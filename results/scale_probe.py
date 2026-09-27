#!/usr/bin/env python3
"""§40 E-0 — 실규모(400×300×100) 전환 비용 실측 프로브. **구현이 아니라 측정** (노트북 무변경).

노트북 셀 3 · 4 (시나리오 풀 생성 앞까지)를 격자 크기만 바꿔 exec 하고, 같은 함수로
  ① 장애물 생성 (현재 생성기 그대로 — guard 1000 포함) 시간 · 실제 채움률
  ② 채움률 12% 를 실제로 채운 격자 (같은 상자 분포, guard 없음)
  ③ k 제약 A* (astar_gnarl) — 시작·목표 거리별 시간 · 전개 수 · 메모리
  ④ 이웃 27개 관측 생성 (build_obs_batch) 1회 시간 — 스텝 비용의 주성분
  ⑤ 시나리오 1개의 메모리 (free + pad)
를 잰다. 검증 격자(50×50×20)도 같은 방법으로 재서 비율을 낸다.

사용: python3 results/scale_probe.py --out results/step1_scale_probe.json
"""
import argparse, contextlib, io, json, os, resource, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402


def boot(nx, ny, nz):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag == "3.":
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "<nb:3>", "exec"), ns)
            c = ns["cfg"]; c.NX, c.NY, c.NZ = nx, ny, nz
            c.pretrain_h = False; c.bc_batches = 0; c.h_rank_batches = 0; c.h_rank_metrics = False
        elif tag == "4.":
            src = src[:src.find("CACHE = (")]          # 풀 생성 앞까지만
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "<nb:4>", "exec"), ns)
    return ns


def fill_grid(ns, rng, fill):
    """현재 생성기와 같은 상자 분포 (2~4 × 2~4 × 1~4) 로 채움률 fill 을 실제로 채운다 (guard 없음)."""
    c = ns["cfg"]
    free = np.ones((c.NX, c.NY, c.NZ), bool)
    target = int(fill * free.size)
    while (~free).sum() < target:
        n = max(1000, int((target - (~free).sum()) / 12))
        sx = rng.integers(2, 5, n); sy = rng.integers(2, 5, n); sz = rng.integers(1, 5, n)
        x = rng.integers(0, c.NX - sx + 1); y = rng.integers(0, c.NY - sy + 1); z = rng.integers(0, c.NZ - sz + 1)
        for i in range(n):
            free[x[i]:x[i] + sx[i], y[i]:y[i] + sy[i], z[i]:z[i] + sz[i]] = False
    return free


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def probe_astar(ns, free, rng, dists, max_expand, per):
    pad = ns["make_pad"](free); D = ns["DIRS26"]; free3 = ns["free3"]
    idx = np.argwhere(free); out = []
    for dist in dists:
        got = 0; tries = 0
        while got < per and tries < 400:
            tries += 1
            a = idx[rng.integers(len(idx))]
            # 거리 dist(맨해튼) 근처의 목표
            b = a + rng.integers(-dist, dist + 1, 3)
            b = np.clip(b, 0, np.array(free.shape) - 1)
            md = int(np.abs(b - a).sum())
            if not free[tuple(b)] or abs(md - dist) > dist * 0.25:
                continue
            cand = [i for i in range(26) if free3(pad, a[0] + D[i][0], a[1] + D[i][1], a[2] + D[i][2])]
            if not cand:
                continue
            r0 = rss_mb(); t0 = time.time()
            r = ns["astar_gnarl"](pad, tuple(int(v) for v in a), tuple(int(v) for v in b), int(cand[0]), max_expand=max_expand)
            dt = time.time() - t0
            out.append(dict(target_dist=dist, manhattan=md, ok=r is not None, sec=round(dt, 3),
                            expanded=(r["expanded"] if r else max_expand), n_cells=(r["n_cells"] if r else None),
                            rss_growth_mb=round(rss_mb() - r0, 1)))
            got += 1
            print(f"   A* dist~{dist:4d} md {md:4d} ok {r is not None} {dt:7.2f}s exp {out[-1]['expanded']:,}", flush=True)
    return out


def probe_obs(ns, free, rng, n=200):
    pad = ns["make_pad"](free); idx = np.argwhere(free)
    goal = tuple(int(v) for v in idx[rng.integers(len(idx))])
    bb = ns["build_obs_batch"]; K = ns["K"]
    import inspect
    sig = inspect.signature(bb)
    ts = []
    for _ in range(n):
        p = tuple(int(v) for v in idx[rng.integers(len(idx))])
        pos = np.array([p] * 27); dirs = np.arange(27) % 26; ss = np.full(27, K); gs = np.zeros(27)
        t0 = time.perf_counter()
        try:
            bb(pad, pos, dirs, ss, gs, goal)
        except TypeError:
            return dict(error=f"signature {sig}")
        ts.append(time.perf_counter() - t0)
    return dict(ms_mean=1000 * float(np.mean(ts)), ms_p90=1000 * float(np.percentile(ts, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-expand", type=int, default=3_000_000)
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    res = {}
    for name, (nx, ny, nz), dists in (("verify", (50, 50, 20), [30, 60]),
                                      ("full", (400, 300, 100), [60, 120, 240])):
        print(f"== {name} {nx}x{ny}x{nz}", flush=True)
        ns = boot(nx, ny, nz); rng = np.random.default_rng(0); R = {}
        t0 = time.time(); free_g = ns["gen_obstacle_grid"](rng)
        R["gen_current"] = dict(sec=round(time.time() - t0, 2), fill=float((~free_g).mean()))
        t0 = time.time(); free_f = fill_grid(ns, rng, ns["cfg"].obstacle_fill)
        R["gen_fill12"] = dict(sec=round(time.time() - t0, 2), fill=float((~free_f).mean()))
        pad = ns["make_pad"](free_f)
        R["scenario_bytes"] = int(free_f.nbytes + pad.nbytes)
        R["J_MAX"] = float(ns["J_MAX"]); R["K"] = int(ns["K"])
        print(f"   gen current {R['gen_current']} · fill12 {R['gen_fill12']} · scen {R['scenario_bytes']/1e6:.1f} MB", flush=True)
        R["obs27"] = probe_obs(ns, free_f, rng); print("   obs27", R["obs27"], flush=True)
        R["astar_fill12"] = probe_astar(ns, free_f, rng, dists, a.max_expand, a.per)
        R["astar_current_gen"] = probe_astar(ns, free_g, rng, dists[-1:], a.max_expand, a.per)
        res[name] = R
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("→", a.out)


if __name__ == "__main__":
    main()
