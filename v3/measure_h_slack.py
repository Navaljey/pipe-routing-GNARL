"""h 느슨함 측정 — admissible 하한 `h` 가 실제 최적 라우터 비용보다 얼마나 낮은가.

소유자 요청 (2026-10-22): "f_min 대비 실제 최적 J 비율 (h 의 느슨함)을 함께 보고한다.
서포트 비중이 큰 배관과 작은 배관으로 나눠서."

배경 — skeleton **D92 / D108**: "남은 서포트 비용의 admissible·consistent 하한" 을 시도했으나
**일관성을 유지하는 하한은 0 뿐**이었다 (표 범위 밖 구간은 라우터 서포트 비용이 0 이고,
서포트 가능 엣지가 10 mm 까지 짧을 수 있다). 확장 수 감소 0% 로 기각.

→ 따라서 `h`(D41 = 남은 유클리드 × kg/m + 최소 꺾임 × e45)는 **서포트 비용과 중력관 계단
비용을 전혀 보지 못한다.** 그 느슨함을 수치로 만드는 것이 이 측정이다.

    looseness = C* / h(start)          C* = 라우터 최적 비용 (직관 + 엘보 + 추정 서포트)
    support_share = J_support_est / C*

최적 비용은 **skeleton 의 공식 컴파일 A\\***(numba, 예산 2²² = D73)로 받는다 — V3 가 다시
구현하지 않는다. skeleton 은 읽기만 한다.

    python3 v3/measure_h_slack.py --set basic --out out/h_slack.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,   # noqa: E402
                                  heuristic_factory, make_layered_router)
from pipe_routing.scenario import load                                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import standard_guard                                          # noqa: E402

SETS = {"basic": ["manual", "procedural"], "manual": ["manual"], "procedural": ["procedural"],
        "stair": ["stair"], "dense": ["dense"]}

# 단일 배관 = 다른 배관 없음·예약 없음 (1단계 E1 단일 배관 설정)
ROUTER = make_layered_router(local_lines=False, support_cost=True, reserve=False,
                             grid=G0, incremental=True, diagnostic=False)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))]


def measure(sc, pipe):
    g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
    h0 = heuristic_factory(g, pipe)(g.start_state(pipe))
    t0 = time.perf_counter()
    r = ROUTER(sc, pipe, 60.0)
    row = {"pipe_id": pipe.id, "size": pipe.nominal_size, "gravity": bool(pipe.gravity_pipe),
           "status": r.status, "h_start": h0, "expanded": r.expanded,
           "sec": round(time.perf_counter() - t0, 2)}
    if r.status == "ok":
        row.update(cost=r.J, J_pipe=r.J_pipe, J_elbow=r.J_elbow, J_support_est=r.J_support_est,
                   bends=r.bends,
                   looseness=(r.J / h0) if h0 > 0 else None,
                   support_share=(r.J_support_est / r.J) if r.J > 0 else None)
    return row


def group_stats(rows, label):
    L = [r["looseness"] for r in rows if r.get("looseness")]
    S = [r["support_share"] for r in rows if r.get("support_share") is not None]
    return {"label": label, "n": len(rows),
            "looseness_median": round(pct(L, 50), 4) if L else None,
            "looseness_p10": round(pct(L, 10), 4) if L else None,
            "looseness_p90": round(pct(L, 90), 4) if L else None,
            "looseness_max": round(max(L), 4) if L else None,
            "support_share_median": round(pct(S, 50), 4) if S else None,
            "expanded_median": pct([r["expanded"] for r in rows], 50)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="basic", choices=sorted(SETS))
    ap.add_argument("--out", default="out/h_slack.json")
    ap.add_argument("--summarize-only", action="store_true",
                    help="이미 쌓인 .jsonl 로 요약만 만든다 (중단 복구용)")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    std = standard_guard.resolve(a.allow_public_standard, what="측정")
    print("  " + standard_guard.banner(std))
    stream = pathlib.Path(a.out).with_suffix(".jsonl")   # 증분 기록 — OOM/중단에도 남는다

    files = []
    for d in SETS[a.set]:
        files += sorted((SK / "scenarios" / d).glob("*.json"))

    t0 = time.perf_counter()
    # 이미 쌓인 .jsonl 은 항상 읽는다 — 재개의 근거다. 안 읽으면 중복이 쌓인다
    by_key = {}
    if stream.exists():
        for line in stream.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_key[(r["scenario"], r["pipe_id"])] = r   # 뒤의 것이 이긴다
        print(f"  .jsonl 에서 {len(by_key)} 라우트 복구")
    rows = list(by_key.values())
    done = set(by_key)
    for f in [] if a.summarize_only else files:
        sc = load(f)
        for p in sc.pipes:
            if (f.stem, p.id) in done:
                continue
            row = measure(sc, p)
            row["scenario"] = f.stem
            rows.append(row)
            with stream.open("a", encoding="utf-8") as fh:        # 한 줄 쓰고 flush
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {f.stem:12s} {p.id:5s} {p.nominal_size:>5s} "
                  f"{'T5' if row['gravity'] else 'T1'} {row['status']:16s} "
                  f"h0={row['h_start']:9.1f} "
                  f"C*={row.get('cost', float('nan')):9.1f} "
                  f"느슨={row.get('looseness', float('nan')):6.2f}x "
                  f"서포트={100 * (row.get('support_share') or 0):5.1f}% ({row['sec']:.1f}s)",
                  flush=True)
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r["status"] == "ok"]
    shares = sorted(r["support_share"] for r in ok)
    cut = pct(shares, 50) if shares else 0.0
    lo = [r for r in ok if r["support_share"] <= cut]
    hi = [r for r in ok if r["support_share"] > cut]
    t1 = [r for r in ok if not r["gravity"]]
    t5 = [r for r in ok if r["gravity"]]

    out = {"set": a.set, "skeleton_commit": os.popen(f"git -C {SK} rev-parse HEAD").read().strip(),
           "grid": "G0", "support_standard": std, "expand_budget": EXPAND_BUDGET,
           "setting": "단일 배관 (placed=(), reserve=False), support_cost=True (D57), skeleton 컴파일 A*",
           "definition": {"looseness": "C* / h(start) — C* = 라우터 최적 비용(직관+엘보+추정 서포트)",
                          "support_share": "J_support_est / C*"},
           "wall_sec": round(wall, 1),
           "n": len(rows), "n_ok": len(ok),
           "n_budget_exhausted": sum(1 for r in rows if r["status"] == "timeout"),
           "n_unreachable": sum(1 for r in rows if r["status"] == "unreachable"),
           "support_share_cut": round(cut, 4),
           "groups": [group_stats(ok, "전체 ok"), group_stats(lo, f"서포트 비중 ≤ {cut:.1%}"),
                      group_stats(hi, f"서포트 비중 > {cut:.1%}"),
                      group_stats(t1, "T1 압력관"), group_stats(t5, "T5 중력관")],
           "routes": rows}
    out["note_oom"] = ("라우트 1개 이상이 OOM·중단으로 빠졌을 수 있다 — n 을 확인할 것. "
                       "원자료는 .jsonl 에 증분 기록된다")
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n라우트 {out['n']} (ok {out['n_ok']} · 예산소진 {out['n_budget_exhausted']} · "
          f"도달불가 {out['n_unreachable']})  {wall:.0f}s")
    print(f"{'그룹':<22} {'n':>4} {'느슨함 중앙':>11} {'p10':>7} {'p90':>7} "
          f"{'서포트비중':>10} {'확장 중앙':>10}")
    for gp in out["groups"]:
        print(f"{gp['label']:<22} {gp['n']:>4} {gp['looseness_median'] or 0:>10.2f}x "
              f"{gp['looseness_p10'] or 0:>6.2f}x {gp['looseness_p90'] or 0:>6.2f}x "
              f"{100 * (gp['support_share_median'] or 0):>9.1f}% {gp['expanded_median'] or 0:>10,}")
    if ok:
        print(f"\n라우터 비용 합계 분해 (ok {len(ok)}): "
              f"직관 {sum(r['J_pipe'] for r in ok):,.0f} · "
              f"엘보 {sum(r['J_elbow'] for r in ok):,.0f} · "
              f"추정 서포트 {sum(r['J_support_est'] for r in ok):,.0f} kg "
              f"({100 * sum(r['J_support_est'] for r in ok) / sum(r['cost'] for r in ok):.1f}%)")
    print(f"→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
