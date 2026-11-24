"""컴파일 A\\* + 노드별 휴리스틱 표 (설계안 §30.3 — 벽시계 판정용).

skeleton `astar_fast._search_generic` 의 **소스를 그대로 가져와 두 곳만 고친다** (skeleton 레포는 읽기 전용):

  ① 인자에 `htab` (노드별 하한, kg) 추가
  ② 휴리스틱 = w · max(skeleton h, htab[노드])  (w = 1 이 기본)

나머지(상태·비용·지배·엘보 규칙·큐 순서)는 한 글자도 바꾸지 않으므로 `htab ≡ 0` 이면 skeleton 과
**확장 수까지 같아야 한다** (시험 `v3/test_astar_rg.py`). `htab` 이 admissible·consistent 하면(완화 그래프 거리
`measure_controls.relaxed_dist`) 결과는 같은 그래프 위 **최적**이다.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import sys
import textwrap

_SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(_SK) not in sys.path:
    sys.path.insert(0, str(_SK))
from pipe_routing import astar_fast as _af                              # noqa: E402


def _patched(fn, edits: list[tuple[str, str]], new_name: str):
    src = textwrap.dedent(inspect.getsource(fn.py_func if hasattr(fn, "py_func") else fn))
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("@"))   # 데코레이터 제거
    for old, new in edits:
        assert src.count(old) == 1, f"skeleton 소스가 바뀌었다 — 패치 지점을 다시 확인: {old[:60]!r}"
        src = src.replace(old, new)
    src = src.replace(f"def {fn.__name__ if not hasattr(fn, 'py_func') else fn.py_func.__name__}(",
                      f"def {new_name}(", 1)
    ns = dict(vars(_af))
    exec(compile(src, f"<v3.astar_rg:{new_name}>", "exec"), ns)
    return ns[new_name], ns


def _build_patch():
    _HLINE = "f = ng + (dist * kgmm + bends * e45)"
    _src0 = textwrap.dedent(inspect.getsource(_af._search_generic.py_func))
    _IND = next(l for l in _src0.splitlines() if l.strip() == _HLINE)[:-len(_HLINE)]   # 들여쓰기를 소스에서 읽는다
    _search_rg_py, _ = _patched(_af._search_generic, [
        ("def _search_generic(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc):",
         "def _search_generic(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc, htab, wgt):"),
        (_IND + _HLINE + "\n",
         _IND + "hh = dist * kgmm + bends * e45\n"
         + _IND + "hr = htab[nflat]\n"
         + _IND + "if hr > hh:\n"
         + _IND + "    hh = hr\n"
         + _IND + "f = ng + wgt * hh\n"),
    ], "_search_rg")
    _search_rg = _af.njit(cache=False)(_search_rg_py)

    _route_py, _ns = _patched(_af.astar_route_generic, [
        ("def astar_route_generic(space, pipe, time_limit, h_start, expand_limit=None):",
         "def astar_route_generic(space, pipe, time_limit, h_start, expand_limit=None, htab=None, wgt=1.0):"),
        ("code = _search_generic(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc)",
         "code = _search_rg(T, C, G, ctr, fctr, H, S, K, open_head, closed_head, arc, htab, float(wgt))"),
    ], "astar_route_rg")
    _ns["_search_rg"] = _search_rg
    astar_route_rg = _route_py
    return astar_route_rg


# D126 (skeleton PR #52) 이후에는 소스 패치 대신 공식 인자 `route_graph(extra_cost, h_lower)` 를 쓴다 (route_api).
# 패치는 D126 이전 skeleton 에서만 만든다 — 이후 소스에서는 패치 지점이 바뀌어 만들지 않는다.
HAVE_D126 = hasattr(__import__('pipe_routing.layered', fromlist=['x']), 'route_graph')
astar_route_rg = None if HAVE_D126 else _build_patch()


def route_rg(g, pipe, htab, expand_limit, w=1.0):
    """→ (status, chain, expanded, generated). h_start 도 같은 max·같은 가중으로.

    w > 1 이면 가중 A* (f = g + w·max(h, htab)) — 비용 ≤ w·C* (재개방 없이도 성립, 지배 가지치기는 그대로)."""
    import numpy as np
    from pipe_routing.layered import heuristic_factory
    h = heuristic_factory(g, pipe)
    s0 = g.start_state(pipe)
    h0 = w * max(h(s0), float(htab[s0.node]))
    if astar_route_rg is None:
        raise RuntimeError("D126 이후 skeleton — route_rg(소스 패치) 대신 route_api 를 쓴다")
    return astar_route_rg(g, pipe, 1e9, h0, expand_limit, np.ascontiguousarray(htab, dtype=np.float64), w)


_BIG = 1e30   # D126 h_lower 는 유한해야 한다 — rg = inf (목표에 못 가는 노드) 는 아주 큰 유한값으로


def route_api(g, pipe, htab, expand_limit, w=1.0, extra_cost=None):
    """D126 공식 호출 — skeleton `layered.route_graph(extra_cost, h_lower)`. → RouteResult (J 는 지도 비용, 가격 제외).

    h_lower = w · rg  →  h = max(D41 h, w · rg).  ⚠️ 소스 패치판 (route_rg) 의 h = w · max(D41 h, rg) 와 **같지 않다**
    (D41 h > rg 인 상태에서 다르다 — 설계안 §38). extra_cost = 협상 가격 (N, 18) ≥ 0, 지도는 바꾸지 않는다."""
    import numpy as np
    from pipe_routing.layered import route_graph
    hl = np.where(np.isfinite(htab), np.asarray(htab, dtype=np.float64) * w, _BIG)
    return route_graph(g, pipe, 1e9, expand_limit, extra_cost=extra_cost, h_lower=hl)
