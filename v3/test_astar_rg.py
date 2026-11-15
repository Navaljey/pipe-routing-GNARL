"""v3.astar_rg — htab ≡ 0 이면 skeleton 컴파일 A* 와 확장 수·비용이 같고, rg 표면 최적 비용이 같다."""
import os, pathlib, sys
SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
sys.path.insert(0, str(SK)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from pipe_routing.generator import GeneratorConfig, generate
from pipe_routing.layered import G0, LayeredGraph, _route
from pipe_routing.router_astar import _finish_chain
from v3.astar_rg import route_rg
from v3.measure_controls import relaxed_dist

def _cases():
    sc = generate(GeneratorConfig(seed=40000000, n_pipes=10), "valid_40000000")
    return sc, [p for p in sc.pipes if p.id in ("P002", "P004", "P007", "P010")]

def test_zero_table_matches_skeleton_and_rg_optimal():
    sc, pipes = _cases()
    for p in pipes:
        g = LayeredGraph(sc, p, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
        ref = _route(g, p, 1e9, 1 << 22)
        st, ch, ex, gen = route_rg(g, p, np.zeros(len(g.pos)), 1 << 22)
        assert st == ref.status and ex == ref.expanded, (p.id, ex, ref.expanded)
        dist, _ = relaxed_dist(g, p)
        st2, ch2, ex2, _ = route_rg(g, p, dist, 1 << 22)
        rr = _finish_chain(g, p, ch2, ex2, 0, 0.0)
        assert st2 == "ok" and abs(rr.J - ref.J) < 1e-6 * ref.J, (p.id, rr.J, ref.J)
        assert ex2 <= ex
        print(p.id, ref.expanded, ex2, round(ref.J, 3))

if __name__ == "__main__":
    test_zero_table_matches_skeleton_and_rg_optimal()
