#!/usr/bin/env python3
"""중력관 구배 규칙 후보의 판단 자료 (§36) — **구현이 아니라 측정**. 노트북·환경은 바꾸지 않는다.

현재 시나리오 풀(학습 500 · 평가 48)에서, 구배 규칙마다 k 제약 A* 를 다시 풀어
① 중력관으로 성립하는 시나리오 비율 ② 압력관 최적 대비 J 증가 ③ 최적 경로가 갈리는 비율 을 잰다.
흐름 방향 = 시작 → 목표 (시작이 상류). z 는 격자 인덱스 (dz=+1 = 오르막).

규칙 (G0 ⊇ G2 ⊇ G1 — 뒤로 갈수록 엄격):
  P   압력관 (현재 환경 그대로)
  G0  오르막 금지 (dz=+1 이동 마스크). 상태 추가 없음
  G1  G0 + **층별 수평 주행 상한**: 마지막 하강(dz=−1 이동) 뒤 같은 z 층에서의 수평 길이 ≤ L = 1/구배 칸.
      50mm 격자에서 수평 칸 L 개 = 실제 관이 구배 s 로 s·L 칸 하강 → 한 칸(50mm) 안에 숨는 한계가 L = 1/s.
      상태 채널 1개 (마지막 하강 뒤 수평 길이) 필요
  G2  G0 + **경로 평균 구배**: 전체 수평 길이 H ≤ (Δz + 1)·L  (Δz = z_시작 − z_목표, +1 은 G1 과 같은 한 칸 양자화 여유).
      G1 을 만족하면 G2 도 만족한다. 상태 채널 1개 (누적 수평 길이 또는 남은 예산) + 남은 거리 하한 lookahead

시작/목표 교체 (swap): 목표가 시작보다 높은 시나리오는 G0 에서 불가능하다. 흐름을 뒤집어(목표→시작) 성립하는지도 잰다.
  교체 시 시작 방향은 생성기와 같은 규칙 (첫 칸이 빈 방향을 셔플, 앞 max_start_dirs 개 중 압력관 A* 가 풀리는 첫 방향).

사용 (_ab/_cache 에서): python3 ../../results/slope_feasibility.py --out ../../results/step1_slope_feasibility.json
"""
import argparse, contextlib, heapq, io, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB  # noqa: E402

NS = None
RULES = [("P", None), ("G0", None), ("G2", 100), ("G2", 50), ("G2", 25), ("G1", 100), ("G1", 50), ("G1", 25)]
# 1/25 는 100A 의 물리 후보가 아니라 **감도** — 50mm 격자에서 규칙이 언제부터 경로를 바꾸는지 보려고 둔다


def build_ns():
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in ("3.", "4."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                ns["cfg"].pretrain_h = False; ns["cfg"].bc_batches = 0
    return ns


def hdist(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def astar_rule(pad, start, goal, start_dir, rule, L, max_expand=3_000_000):
    """노트북 astar_gnarl 과 같은 전이·비용 + 구배 규칙. 상태 = (pos, dir, s, c).
    c: G1 = 마지막 하강 뒤 수평 길이, G2 = 누적 수평 길이, 그 밖 0.
    같은 (pos, dir, s) 에서 c 가 작은 쪽이 지배하므로 (h 는 pos 만의 함수라 같은 키는 g 순서로 꺼내진다)
    닫힌 최소 c 이상인 상태는 버린다 — 정확한 가지치기."""
    ns = NS
    K, DIRS26, SPEC = ns["K"], ns["DIRS26"], ns["SPEC"]
    mask, move_J, hfn = ns["action_mask_27"], ns["move_J"], ns["h_octile3d_J"]
    start = tuple(int(v) for v in start); goal = tuple(int(v) for v in goal)
    budget = (start[2] - goal[2] + 1) * L if rule == "G2" else None
    if rule in ("G0", "G1", "G2") and goal[2] > start[2]:
        return dict(status="infeasible_dz")
    st0 = (start, int(start_dir), K, 0.0)
    g = {st0: 0.0}; came = {}; cnt = 0; expanded = 0; cmin = {}
    heap = [(hfn(start, goal), 0, st0)]
    while heap:
        f, _, st = heapq.heappop(heap)
        pos, d, s, c = st
        key = (pos, d, s)
        if key in cmin and cmin[key] <= c + 1e-9:
            continue
        cmin[key] = c
        if pos == goal:
            path = [st]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return dict(status="ok", J=g[st], expanded=expanded, path=[p[0] for p in path])
        expanded += 1
        if expanded > max_expand:
            return dict(status="limit", expanded=expanded)
        m = mask(pad, pos, d, s, goal, K)
        for i in np.where(m[:26])[0]:
            i = int(i); dx, dy, dz = DIRS26[i]
            if rule != "P" and dz > 0:
                continue
            hstep = math.hypot(dx, dy)
            if rule == "G1":
                nc = 0.0 if dz < 0 else c + hstep
                if nc > L + 1e-9:
                    continue
            elif rule == "G2":
                nc = c + hstep
            else:
                nc = 0.0
            npos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if rule == "G2" and nc + hdist(npos, goal) > budget + 1e-9:
                continue
            nc = round(nc, 6)
            dJ, _, _ = move_J(i, d, SPEC)
            nsv = min(s + 1, K) if i == d else 1
            nst = (npos, i, nsv, nc)
            ng = g[st] + dJ
            if ng < g.get(nst, math.inf) - 1e-12:
                g[nst] = ng; came[nst] = st; cnt += 1
                heapq.heappush(heap, (ng + hfn(npos, goal), cnt, nst))
    return dict(status="infeasible")


def path_stats(path):
    """압력관 경로가 중력 규칙을 얼마나 어기는가: 오르막 이동 수 · 층별 최대 수평 주행 · 누적 수평 길이."""
    climbs, run, maxrun, H = 0, 0.0, 0.0, 0.0
    for a, b in zip(path[:-1], path[1:]):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        h = math.hypot(dx, dy); H += h
        if dz > 0: climbs += 1; run = 0.0
        elif dz < 0: run = 0.0
        else: run += h; maxrun = max(maxrun, run)
    return dict(climbs=climbs, maxrun=round(maxrun, 3), H=round(H, 3))


def swap_dir(sc, idx):
    """교체 시 시작 방향 — 생성기와 같은 규칙 (셔플은 시나리오 번호 시드)."""
    ns = NS
    pad, a, b = sc["pad"], tuple(sc["goal"]), tuple(sc["start"])
    D, free3 = ns["DIRS26"], ns["free3"]
    cand = [i for i in range(26) if free3(pad, a[0] + D[i][0], a[1] + D[i][1], a[2] + D[i][2])]
    rng = np.random.default_rng(10_000 + idx)
    rng.shuffle(cand)
    for d in cand[:max(1, ns["cfg"].max_start_dirs)]:
        r = astar_rule(pad, a, b, d, "P", None)
        if r["status"] == "ok":
            return d, r
    return None, None


def one(job):
    pool, idx = job
    sc = (NS["TRAIN_POOL"] if pool == "train" else NS["EVAL_POOL"])[idx]
    out = dict(pool=pool, idx=idx, start=list(sc["start"]), goal=list(sc["goal"]),
               dz=int(sc["start"][2] - sc["goal"][2]))
    t0 = time.time()
    for tag, a, b, d0 in (("orig", sc["start"], sc["goal"], sc["dirs"][0]),
                          ("swap", sc["goal"], sc["start"], None)):
        res = {}
        if tag == "swap":
            d0, rp = swap_dir(sc, idx)
            if d0 is None:
                out[tag] = dict(status="no_start_dir"); continue
        for rule, L in RULES:
            name = rule if L is None else f"{rule}_{L}"
            r = astar_rule(sc["pad"], a, b, d0, rule, L)
            if r["status"] == "ok":
                if rule == "P":
                    res["P_stats"] = path_stats(r["path"])
                r.pop("path")
            res[name] = r
        if tag == "orig":   # 건전성: 압력관 A* 가 캐시의 기준선 J 와 같아야 한다
            res["P_cache_J"] = float(sc["astar"][sc["dirs"][0]]["J"])
        out[tag] = res
    out["sec"] = round(time.time() - t0, 2)
    return out


def main():
    global NS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    NS = build_ns()
    jobs = [("eval", i) for i in range(len(NS["EVAL_POOL"]))] + \
           [("train", i) for i in range(len(NS["TRAIN_POOL"]))]
    if a.limit:
        jobs = jobs[:a.limit]
    t0 = time.time(); rows = []
    with Pool(a.procs) as p:
        for k, r in enumerate(p.imap_unordered(one, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 50 == 0:
                print(f"{k+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["pool"], r["idx"]))
    json.dump(dict(rules=[r if l is None else f"{r}_{l}" for r, l in RULES], rows=rows,
                   wall_sec=round(time.time() - t0, 1)), open(a.out, "w"), indent=1)
    print("done", a.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
