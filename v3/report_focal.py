"""|FOCAL| 측정 원자료 → 설계안 §9 표. **평균 하나로 뭉개지 않는다** (D-v).

읽는 것: `measure_focal.py` 의 `.jsonl` (라우트당 한 줄, 표본 목록 포함).
내는 것: ① ε 별 분포 ② **탐색 깊이별** ③ 구경별 ④ `C*/f_min` (h 느슨함의 탐색 중 판)
⑤ 측정 한계 (지배검사 생략 비율·예산 소진·메모리).

왜 깊이별로 보나: 1차 측정에서 **쉬운 라우트엔 띠에 여지가 없고 어려운 라우트엔
매우 컸다.** 이득이 필요한 곳에 여지가 있다는 뜻이라 유리한 모양이지만, 전체
중앙 하나로 적으면 그 구조가 가려진다.

    python3 v3/report_focal.py out/focal_basic_priv.jsonl
"""

from __future__ import annotations

import collections
import json
import statistics
import sys


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))]


def fmt(x, w=9, d=0):
    return "—".rjust(w) if x is None else f"{x:>{w},.{d}f}"


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[(r["scenario"], r["pipe_id"])] = r      # 뒤의 것이 이긴다
    return list(rows.values())


def eps_keys(rows):
    for r in rows:
        if r.get("samples"):
            return sorted(r["samples"][0]["focal"], key=float)
    return []


def band(rows, keys, pick, label):
    """pick(sample) 이 참인 표본만 모아 ε 별 한 줄."""
    out = []
    for e in keys:
        f = [s["focal"][e] for r in rows for s in r.get("samples", ()) if pick(s)]
        v = [s["open_viable"] for r in rows for s in r.get("samples", ()) if pick(s)]
        if not f:
            continue
        out.append((e, len(f), pct(f, 50), pct(f, 90), max(f),
                    sum(1 for x in f if x <= 1) / len(f),
                    statistics.fmean(f) / statistics.fmean(v) if v and statistics.fmean(v) else None))
    if out:
        print(f"\n  {label}  (표본 {out[0][1]:,})")
        print(f"  {'ε':>6} {'중앙':>8} {'p90':>9} {'최대':>9} {'=1 비율':>9} {'viable 대비':>11}")
        for e, _, med, p90, mx, eq1, ratio in out:
            print(f"  {e:>6} {fmt(med, 8)} {fmt(p90, 9)} {fmt(mx, 9)} {eq1:>8.1%} "
                  f"{(f'{ratio:.1%}' if ratio else '—'):>11}")


def main(argv):
    path = argv[1] if len(argv) > 1 else "out/focal_basic_priv.jsonl"
    rows = load(path)
    keys = eps_keys(rows)
    st = collections.Counter(r["status"] for r in rows)
    samples = sum(len(r.get("samples", ())) for r in rows)
    print(f"원자료 {path}")
    print(f"라우트 {len(rows)}  표본 {samples:,}  상태 {dict(st)}")
    exp = [r["expanded"] for r in rows]
    print(f"확장 중앙 {pct(exp, 50):,} · p90 {pct(exp, 90):,} · 최대 {max(exp):,}")

    band(rows, keys, lambda s: True, "① 전체 표본")
    band(rows, keys, lambda s: s["expanded"] <= 1_000, "② 탐색 초반 (확장 ≤ 1,000)")
    band(rows, keys, lambda s: 1_000 < s["expanded"] <= 10_000, "③ 중반 (1,000 < 확장 ≤ 10,000)")
    band(rows, keys, lambda s: s["expanded"] > 10_000, "④ 후반 (확장 > 10,000)")

    # 구경별 — 라우트 단위 중앙의 중앙 (라우트 수 가중을 피한다)
    e01 = "0.01" if "0.01" in keys else (keys[len(keys) // 2] if keys else None)
    if e01:
        print(f"\n  ⑤ 구경별 (ε={e01}, 라우트별 중앙의 중앙)")
        print(f"  {'구경':>6} {'라우트':>6} {'|F| 중앙':>9} {'확장 중앙':>11} {'ok':>4}")
        by = collections.defaultdict(list)
        for r in rows:
            by[r["size"]].append(r)
        for size in sorted(by, key=lambda s: int(s.rstrip("A"))):
            rs = by[size]
            meds = [statistics.median([s["focal"][e01] for s in r["samples"]])
                    for r in rs if r.get("samples")]
            print(f"  {size:>6} {len(rs):>6} {fmt(pct(meds, 50), 9)} "
                  f"{fmt(pct([r['expanded'] for r in rs], 50), 11)} "
                  f"{sum(1 for r in rs if r['status'] == 'ok'):>4}")

    # C*/f_min — 탐색 중 하한이 얼마나 낮은가 (§10 느슨함의 탐색 중 판)
    ok = [r for r in rows if r["status"] == "ok" and r.get("samples") and r.get("cost")]
    if ok:
        start = [r["cost"] / r["samples"][0]["f_min"] for r in ok if r["samples"][0]["f_min"] > 0]
        last = [r["cost"] / r["samples"][-1]["f_min"] for r in ok if r["samples"][-1]["f_min"] > 0]
        allr = [r["cost"] / s["f_min"] for r in ok for s in r["samples"] if s["f_min"] > 0]
        print(f"\n  ⑥ C*/f_min — 하한이 최적 비용보다 얼마나 낮은가 (ok {len(ok)})")
        print(f"  {'시점':>10} {'중앙':>8} {'p90':>8} {'최대':>8}")
        for lab, xs in (("첫 표본", start), ("마지막 표본", last), ("전 표본", allr)):
            print(f"  {lab:>10} {fmt(pct(xs, 50), 8, 3)} {fmt(pct(xs, 90), 8, 3)} "
                  f"{fmt(max(xs) if xs else None, 8, 3)}")
        print("  (1.0 에 가까울수록 탐색 끝에서 하한이 조여진다 — 띠의 절대 폭 해석에 쓴다)")

    # 측정 한계
    nodom = sum(1 for r in rows for s in r.get("samples", ()) if not s["dom_checked"])
    print(f"\n  ⑦ 한계: 지배검사 생략 표본 {nodom:,} / {samples:,} ({nodom / max(1, samples):.1%}) · "
          f"예산 소진 {st.get('budget_exhausted', 0)} · 메모리 {st.get('memory_exhausted', 0)} · "
          f"도달불가 {st.get('unreachable', 0)}")
    print("  전개 순서는 순수 A\\* 다 — 포컬 탐색 자체 순서에서 잰 값과는 다를 수 있다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
