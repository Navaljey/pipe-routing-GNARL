"""포컬 탐색 등가성 시험 — ε=0 이면 skeleton A\\* 와 **같은 것**이어야 한다.

왜 이 시험이 필요한가: `v3/focal.py` 는 skeleton 의 파이썬 A\\* 를 다시 쓴 코드다
(FOCAL 선택을 끼워 넣을 자리가 필요했다). 다시 쓴 코드는 미묘하게 다를 수 있고,
다르면 1단계 측정의 **기준선 비교가 전부 무의미**해진다 — "확장 수가 줄었다" 가
구현 차이일 수 있기 때문이다.

그래서 ε=0 · priority=None 에서 세 가지가 **정확히** 같아야 한다:

    ① 경로 (상태 사슬의 노드·방향)
    ② 라우터 비용 g (직관 + 엘보 + 추정 서포트 + 관통)
    ③ **확장 수** ← 이게 핵심이다. 1단계 승리 조건의 측정 대상이다

skeleton 쪽은 `impl="python"` 으로 고정한다 — 컴파일 구현(numba)과 파이썬 구현은
skeleton 자체 시험에서 이미 비트 동일이고(D53), 우리가 비교해야 할 것은 전개 순서다.

    PIPE_SUPPORT_STANDARD=... python3 v3/test_focal_equivalence.py     # pytest 없이도 실행된다
    PIPE_SUPPORT_STANDARD=... pytest v3/test_focal_equivalence.py
"""

from __future__ import annotations

import os
import pathlib
import sys

try:
    import pytest
except ModuleNotFoundError:        # 이 컨테이너엔 pytest 가 없다 — 직접 실행 경로를 남긴다
    pytest = None

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pipe_routing.layered import G0, LayeredGraph, heuristic_factory   # noqa: E402
from pipe_routing.router_astar import astar_route                      # noqa: E402
from pipe_routing.scenario import load                                 # noqa: E402

from v3.focal import focal_search                                      # noqa: E402

LIMIT = 200_000          # 시험이 오래 걸리지 않게. 예산 소진도 비교 대상이다
TOL = 1e-6               # 같은 부동소수 연산 순서라 사실상 0 이어야 한다


def _graph(sc, pipe):
    """1단계 단일 배관 설정 — 다른 배관 없음·예약 없음."""
    return LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)


def _cases(set_name: str = "manual", n_scenarios: int = 0):
    out = []
    files = sorted((SK / "scenarios" / set_name).glob("*.json"))
    for f in files[:n_scenarios] if n_scenarios else files:
        sc = load(f)
        for p in sc.pipes:
            out.append((f.stem, sc, p))
    return out


def compare(sc, pipe, limit: int = LIMIT):
    """skeleton A\\* 와 ε=0 포컬 탐색을 같은 그래프에서 돌려 맞춰 본다.

    예산을 작게 줘도 의미가 있다 — 둘이 **같은 확장 수에서** 소진되면 그 지점까지
    전개 순서가 같았다는 뜻이다.
    """
    g1 = _graph(sc, pipe)
    ref = astar_route(g1, pipe, time_limit=1e9, impl="python", expand_limit=limit)

    g2 = _graph(sc, pipe)
    got = focal_search(g2, pipe, heuristic_factory(g2, pipe), eps=0.0, expand_limit=limit)

    # skeleton 은 예산 소진을 "timeout" 으로 쓴다 (D72 — 같은 뜻)
    ref_status = "budget_exhausted" if ref.status == "timeout" else ref.status
    assert got.status == ref_status, f"{pipe.id}: 상태 {got.status} ≠ {ref_status}"
    assert got.expanded == ref.expanded, \
        f"{pipe.id}: 확장 수 {got.expanded:,} ≠ {ref.expanded:,} — 전개 순서가 다르다"
    if ref_status == "ok":
        assert abs(got.cost - ref.J) <= TOL * max(1.0, abs(ref.J)), \
            f"{pipe.id}: 비용 {got.cost} ≠ {ref.J}"
        # 경로 전체를 비교한다 — 꺾임점(waypoints)까지 같아야 한다
        pts = [g2.position(s.node) for s in got.states]
        waypoints = [tuple(pts[0])]
        for i in range(1, len(got.states)):
            if got.states[i - 1].dir != got.states[i].dir:
                waypoints.append(tuple(pts[i - 1]))
        waypoints.append(tuple(pts[-1]))
        ref_wp = [tuple(w) for w in ref.waypoints]
        assert waypoints == ref_wp, \
            f"{pipe.id}: 꺾임점 {len(waypoints)} ≠ {len(ref_wp)} 또는 좌표 불일치"
    return ref, got


if pytest is not None:
    @pytest.mark.parametrize("name,sc,pipe", [(n, s, p) for n, s, p in _cases()],
                             ids=[f"{n}:{p.id}" for n, _, p in _cases()])
    def test_focal_eps0_matches_skeleton_astar(name, sc, pipe):
        compare(sc, pipe)


if __name__ == "__main__":
    import argparse

    from v3 import standard_guard
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="manual")
    ap.add_argument("--scenarios", type=int, default=0, help="앞에서 N개만 (0 = 전부)")
    ap.add_argument("--limit", type=int, default=LIMIT,
                    help="확장 상한. 작게 주면 '같은 지점에서 소진'을 본다")
    ap.add_argument("--ok-under", type=int, default=0, metavar="N",
                    help="측정 .jsonl 에서 ok 로 끝났고 확장 ≤ N 인 라우트만 고른다 "
                         "— 경로·비용까지 비교되는 사례를 싸게 모은다")
    ap.add_argument("--jsonl", default="out/focal_basic_priv.jsonl")
    a = ap.parse_args()
    std = standard_guard.resolve(False, what="등가성 시험")
    print("  " + standard_guard.banner(std))
    cases = _cases(a.set, a.scenarios)
    if a.ok_under:
        import json
        want = set()
        for line in pathlib.Path(a.jsonl).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["status"] == "ok" and r["expanded"] <= a.ok_under:
                    want.add((r["scenario"], r["pipe_id"]))
        cases = [c for c in cases if (c[0], c[2].id) in want]
        print(f"  .jsonl 에서 ok·확장 ≤ {a.ok_under:,} 라우트 {len(cases)} 선택")

    bad = 0
    for name, sc, pipe in cases:
        try:
            ref, got = compare(sc, pipe, a.limit)
            print(f"  ✅ {name:12s} {pipe.id:5s} {pipe.nominal_size:>5s} {got.status:17s} "
                  f"확장 {got.expanded:>8,d}  비용 {got.cost if got.cost else float('nan'):10.2f}")
        except AssertionError as e:
            bad += 1
            print(f"  ❌ {name:12s} {pipe.id:5s} {e}")
    print(f"\n{'전부 일치' if not bad else f'불일치 {bad} 건'}")
    raise SystemExit(1 if bad else 0)
