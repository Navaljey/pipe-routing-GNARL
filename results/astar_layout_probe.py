#!/usr/bin/env python3
"""§42-3-1 — 실규모(400×300×100, 채움 12%) A* 비용의 장애물 분포 의존 재확인. **측정만** (노트북 무변경).
조건은 CLAUDE.md §42-3-1 에 실행 전 고정. octile 휴리스틱 (현재 기본) 만 쓴다.

격자 G 개 (rng 시드 grid_seed0 + i) × 목표 거리마다 per 쌍. 쌍은 장애물만 봐도 연결된 것만 쓴다
(목표에서의 장애물 인식 거리장이 시작점에서 유한 — 끊긴 쌍은 뽑기에서 버리고 수만 센다).
A* 실패 (None) = 전개 한도 도달 또는 k 제약상 도달 불가 → 전개 수는 한도로 기록 (하한).

사용: python3 results/astar_layout_probe.py --out results/X.json [--box 20,20,10,60,60,40]
"""
import argparse, contextlib, io, json, math, os, resource, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB


def boot():
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag == "3.":
            assert src.count("cfg = Step1Config()") == 1
            src = src.replace("cfg = Step1Config()", "cfg = Step1Config(NX=400, NY=300, NZ=100)")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "<nb:3>", "exec"), ns)
            ns["cfg"].pretrain_h = False; ns["cfg"].bc_batches = 0
        elif tag == "4.":
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src[:src.find("CACHE = (")], "<nb:4>", "exec"), ns)
    return ns


def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--box", default=None, help="상자 크기 'lx,ly,lz,hx,hy,hz' (칸). 없으면 생성기 기본 (2~4 × 2~4 × 1~4)")
    ap.add_argument("--grids", type=int, default=4); ap.add_argument("--grid-seed0", type=int, default=100)
    ap.add_argument("--per", type=int, default=3); ap.add_argument("--dists", default="270,500")
    ap.add_argument("--max-expand", type=int, default=6_000_000)
    a = ap.parse_args()
    ns = boot(); cfg = ns["cfg"]; cfg.astar_heuristic = "octile"
    if a.box:
        v = [int(x) for x in a.box.split(",")]; cfg.obstacle_box_min, cfg.obstacle_box_max = tuple(v[:3]), tuple(v[3:])
    D = ns["DIRS26"]; dists = [int(x) for x in a.dists.split(",")]
    out = dict(box=a.box, grids=[], rows=[], max_expand=a.max_expand)
    for gi in range(a.grids):
        rng = np.random.default_rng(a.grid_seed0 + gi)
        t0 = time.time(); free = ns["gen_obstacle_grid"](rng); pad = ns["make_pad"](free)
        idx = np.argwhere(free)
        G = dict(grid=gi, seed=a.grid_seed0 + gi, fill=float((~free).mean()), gen_sec=round(time.time() - t0, 1), rejected_disconnected=0)
        out["grids"].append(G); print(json.dumps(G), flush=True)
        for dist in dists:
            got = 0
            while got < a.per:
                s = idx[rng.integers(len(idx))]; g = s + rng.integers(-dist, dist + 1, 3)
                g = np.clip(g, 0, np.array(free.shape) - 1)
                if not free[tuple(g)] or abs(int(np.abs(g - s).sum()) - dist) > dist * 0.15: continue
                cand = [i for i in range(26) if ns["free3"](pad, s[0] + D[i][0], s[1] + D[i][1], s[2] + D[i][2])]
                if not cand: continue
                s, g = tuple(int(v) for v in s), tuple(int(v) for v in g)
                t0 = time.time(); field, _ = ns["geodesic_field"](free, g); fsec = time.time() - t0
                if not math.isfinite(float(field[s])):
                    G["rejected_disconnected"] += 1; continue
                got += 1
                row = dict(grid=gi, target=dist, manhattan=int(sum(abs(x - y) for x, y in zip(s, g))), start=list(s), goal=list(g),
                           start_dir=int(cand[0]), field_sec=round(fsec, 1), geo_cells=float(field[s]),
                           octile_cells=float(ns["h_octile3d_cells"](s, g)))
                del field
                r0 = rss(); t0 = time.time()
                r = ns["astar_gnarl"](pad, s, g, int(cand[0]), max_expand=a.max_expand)
                row.update(ok=r is not None, sec=round(time.time() - t0, 1), J=(r["J"] if r else None),
                           n_cells=(r["n_cells"] if r else None), expanded=(r["expanded"] if r else a.max_expand),
                           rss_growth_mb=round(rss() - r0))
                out["rows"].append(row); print(json.dumps(row), flush=True)
                json.dump(out, open(a.out, "w"), indent=1)
        json.dump(out, open(a.out, "w"), indent=1)
    print("→", a.out)


if __name__ == "__main__":
    main()
