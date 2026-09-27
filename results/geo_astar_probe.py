#!/usr/bin/env python3
"""§42 ③ — 실규모(400×300×100, 12% 상자) 에서 A* 휴리스틱 octile vs geodesic(장애물 인식 거리장) 의 전개 수 · 시간 비교.
같은 시작·목표 쌍에 두 휴리스틱을 짝지어 돌린다. 노트북 셀 3·4 (풀 생성 앞까지)를 격자만 바꿔 exec — 노트북 함수 그대로.
사용: python3 results/geo_astar_probe.py --out results/step1_geo_astar_probe.json [--octile-500 0]"""
import argparse, contextlib, io, json, math, os, resource, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB

def boot():
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag == "3.":
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
    ap.add_argument("--per", type=int, default=3); ap.add_argument("--max-expand", type=int, default=6_000_000)
    ap.add_argument("--octile-500", type=int, default=1, help="거리 500 에서 octile 도 돌릴 쌍 수 (비싸다)")
    ap.add_argument("--per-500", type=int, default=None, help="거리 500 쌍 수 (기본 --per 와 같다)")
    ap.add_argument("--box", default=None, help="민감도용 상자 크기 'lx,ly,lz,hx,hy,hz' (칸). 장애물 모델 제안이 아니다")
    a = ap.parse_args()
    ns = boot(); cfg = ns["cfg"]; rng = np.random.default_rng(7)
    if a.box:
        v = [int(x) for x in a.box.split(",")]; cfg.obstacle_box_min, cfg.obstacle_box_max = tuple(v[:3]), tuple(v[3:])
    free = ns["gen_obstacle_grid"](rng); pad = ns["make_pad"](free); D = ns["DIRS26"]
    idx = np.argwhere(free); out = dict(fill=float((~free).mean()), box=a.box, rows=[])
    for dist in (270, 500):
        got = 0
        while got < (a.per if dist == 270 or a.per_500 is None else a.per_500):
            s = idx[rng.integers(len(idx))]; g = s + rng.integers(-dist, dist + 1, 3)
            g = np.clip(g, 0, np.array(free.shape) - 1)
            if not free[tuple(g)] or abs(int(np.abs(g - s).sum()) - dist) > dist * 0.15: continue
            cand = [i for i in range(26) if ns["free3"](pad, s[0] + D[i][0], s[1] + D[i][1], s[2] + D[i][2])]
            if not cand: continue
            s, g, d0 = tuple(int(v) for v in s), tuple(int(v) for v in g), int(cand[0]); got += 1
            row = dict(target=dist, manhattan=int(sum(abs(x - y) for x, y in zip(s, g))))
            ns["_FIELD_CACHE"].clear()
            t0 = time.time(); field, rounds = ns["geodesic_field"](free, g); row["field_sec"] = round(time.time() - t0, 1); row["field_rounds"] = rounds
            row["geo_cells"] = float(field[s]); row["octile_cells"] = float(ns["h_octile3d_cells"](s, g))
            for heur in ("geodesic", "octile"):
                if heur == "octile" and dist == 500 and got > a.octile_500: continue
                cfg.astar_heuristic = heur; r0 = rss(); t0 = time.time()
                r = ns["astar_gnarl"](pad, s, g, d0, max_expand=a.max_expand)
                row[heur] = dict(ok=r is not None, sec=round(time.time() - t0, 1), J=(r["J"] if r else None),
                                 expanded=(r["expanded"] if r else a.max_expand), rss_growth_mb=round(rss() - r0))
            out["rows"].append(row); print(json.dumps(row), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
    print("→", a.out)

if __name__ == "__main__":
    main()
