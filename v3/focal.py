"""포컬 탐색 (A*_ε, Pearl & Kim 1982) — V3 1단계 라우터 슬롯.

설계안: `docs/design_v3_architecture.md` §2.1.

    OPEN  : f = g + h 순          (h = skeleton heuristic_factory, admissible — D41)
    FOCAL : {s ∈ OPEN : f(s) ≤ (1+ε)·f_min}
    확장  : FOCAL 안에서 **학습 점수** 순 (priority). 없으면 f 순 = 순수 A*

보장: 반환 경로의 **라우터 비용**(직관 + 엘보 + 추정 서포트 D57) ≤ (1+ε) × 그 표현 위
최적. ⚠️ 검증기 J 에 대한 보장이 아니다 — D57 추정 서포트의 오차가 중앙 +18.6% 다
(skeleton D57). 판정은 측정으로 한다 (소유자 회신 3, 2026-10-21).

이 모듈은 skeleton 을 **읽기만** 한다. 고치지 않는다 (계약 §b).
표현은 `skeleton.pipe_routing.layered.LayeredGraph` 를 그대로 쓴다 (D9 인터페이스).

측정 모드
---------
`measure=MeasureSpec(...)` 를 주면 **순수 A\\* 전개 순서**를 유지한 채 표본 확장 시점에서
ε 별 |FOCAL| 을 센다. 포컬 탐색 자체의 전개 순서에서 재는 것과는 다르다 (§주의).
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

EPS = 1e-9


# ---------------------------------------------------------------------------
# 지배 가지치기 — skeleton router_astar.dominated 와 같은 규칙 (D45·D60·D61·D65)
# ---------------------------------------------------------------------------

def _dominated(closed, s, g) -> bool:
    """같은 (node, dir) 에서 run ≥ · bend ≤ · ends ⊇ · hz ≤ · g ≤ 인 확정 상태가 있으면 버린다."""
    for run, bend, ends, hz, gc in closed.get((s.node, s.dir), ()):
        if run >= s.run and bend <= s.bend and (ends & s.ends) == s.ends and hz <= s.hz and gc <= g + EPS:
            return True
    return False


# ---------------------------------------------------------------------------
# |FOCAL| 측정
# ---------------------------------------------------------------------------

@dataclass
class MeasureSpec:
    """어디서 몇 번 재는가. |OPEN| 전수 조사라 표본 간격을 둔다."""
    eps_list: tuple = (0.005, 0.01, 0.02, 0.05)
    every: int = 256            # 이 간격마다 1회
    log_head: int = 64          # 초반 1..log_head 확장은 2의 거듭제곱 시점에서도 잰다
    max_samples: int = 400      # 라우트당 상한 (측정 비용 억제)


@dataclass
class FocalStats:
    """ε → |FOCAL| 표본 목록. viable = 낡지 않고 지배되지 않은 OPEN 항목만 센 것."""
    samples: list = field(default_factory=list)   # [{expanded, open_raw, open_viable, f_min, focal: {eps: n}}]
    n_samples: int = 0
    measure_sec: float = 0.0


@dataclass
class FocalResult:
    status: str                  # ok | unreachable | budget_exhausted
    cost: float | None = None    # 라우터 비용 (직관 + 엘보 + 추정 서포트)
    states: list = field(default_factory=list)
    expanded: int = 0
    generated: int = 0
    search_sec: float = 0.0
    eps: float = 0.0
    stats: FocalStats | None = None


def _count_focal(open_heap, g_best, closed, eps_list, viable_limit):
    """현재 OPEN 에서 ε 별 |FOCAL| 을 센다.

    viable = ① 낡지 않음 (g ≤ g_best[state]) ② 지배되지 않음.
    ②는 비용이 크므로 viable 후보가 viable_limit 이하일 때만 적용하고,
    넘으면 ①만 적용한 값을 쓰고 `dom_checked=False` 로 표시한다.
    """
    fresh = [(f, g, s) for f, _, g, s in open_heap if g <= g_best.get(s, math.inf) + EPS]
    dom_checked = len(fresh) <= viable_limit
    if dom_checked:
        fresh = [(f, g, s) for f, g, s in fresh if not _dominated(closed, s, g)]
    if not fresh:
        return None
    fs = [f for f, _, _ in fresh]
    f_min = min(fs)
    out = {"open_raw": len(open_heap), "open_viable": len(fresh), "f_min": f_min,
           "dom_checked": dom_checked, "focal": {}}
    for e in eps_list:
        thr = f_min * (1.0 + e) if f_min >= 0 else f_min * (1.0 - e)
        out["focal"][e] = sum(1 for f in fs if f <= thr + EPS)
    return out


# ---------------------------------------------------------------------------
# 탐색
# ---------------------------------------------------------------------------

def focal_search(space, pipe, heuristic, *, eps: float = 0.0, priority=None,
                 expand_limit: int | None = None, time_limit: float | None = None,
                 measure: MeasureSpec | None = None) -> FocalResult:
    """eps=0 이고 priority=None 이면 skeleton 파이썬 A\\* 와 같은 전개 순서다.

    priority(state, g, h, f) → 작은 값을 먼저. 학습 ĥ 가 들어갈 자리 (D-p).
    """
    t0 = time.perf_counter()
    h = heuristic
    start = space.start_state(pipe)
    tie = itertools.count()
    open_heap = [(h(start), next(tie), 0.0, start)]
    g_best = {start: 0.0}
    parent = {start: None}
    closed: dict = {}
    expanded = generated = 0
    st = FocalStats() if measure else None
    m_next = 1
    m_sec = 0.0

    while open_heap:
        # ── 측정 (전개 순서를 바꾸지 않는다)
        if measure and st.n_samples < measure.max_samples and expanded >= m_next:
            tm = time.perf_counter()
            row = _count_focal(open_heap, g_best, closed, measure.eps_list, viable_limit=40_000)
            if row is not None:
                row["expanded"] = expanded
                st.samples.append(row)
                st.n_samples += 1
            m_sec += time.perf_counter() - tm
            m_next = (m_next * 2 if expanded < measure.log_head
                      else (expanded // measure.every + 1) * measure.every)

        # ── FOCAL 선택
        if eps > 0.0 and priority is not None:
            f_min = open_heap[0][0]
            thr = f_min * (1.0 + eps) + EPS
            best_i, best_k = None, None
            for i, (f, _, g, s) in enumerate(open_heap):
                if f > thr:
                    continue
                if g > g_best.get(s, math.inf) + EPS or _dominated(closed, s, g):
                    continue
                k = priority(s, g, f - g, f)
                if best_k is None or k < best_k:
                    best_i, best_k = i, k
            if best_i is None:
                heapq.heappop(open_heap)
                continue
            f, _, g, s = open_heap[best_i]
            open_heap[best_i] = open_heap[-1]
            open_heap.pop()
            if best_i < len(open_heap):
                heapq._siftup(open_heap, best_i)
                heapq._siftdown(open_heap, 0, best_i)
        else:
            f, _, g, s = heapq.heappop(open_heap)

        if g > g_best.get(s, math.inf) + EPS or _dominated(closed, s, g):
            continue
        closed.setdefault((s.node, s.dir), []).append((s.run, s.bend, s.ends, s.hz, g))
        expanded += 1

        if space.is_goal(s, pipe):
            chain, cur = [], s
            while cur is not None:
                chain.append(cur)
                cur = parent[cur]
            chain.reverse()
            if st:
                st.measure_sec = m_sec
            return FocalResult("ok", g, chain, expanded, generated,
                               time.perf_counter() - t0 - m_sec, eps, st)

        if expanded % 2048 == 0:
            if expand_limit is not None and expanded >= expand_limit:
                if st:
                    st.measure_sec = m_sec
                return FocalResult("budget_exhausted", None, [], expanded, generated,
                                   time.perf_counter() - t0 - m_sec, eps, st)
            if time_limit is not None and time.perf_counter() - t0 - m_sec > time_limit:
                if st:
                    st.measure_sec = m_sec
                return FocalResult("budget_exhausted", None, [], expanded, generated,
                                   time.perf_counter() - t0 - m_sec, eps, st)

        for nb in space.neighbors(s, pipe):
            ng = g + space.cost(s, nb, pipe)
            if ng + EPS < g_best.get(nb, math.inf) and not _dominated(closed, nb, ng):
                g_best[nb] = ng
                parent[nb] = s
                heapq.heappush(open_heap, (ng + h(nb), next(tie), ng, nb))
                generated += 1

    if st:
        st.measure_sec = m_sec
    return FocalResult("unreachable", None, [], expanded, generated,
                       time.perf_counter() - t0 - m_sec, eps, st)
