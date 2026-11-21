"""협상 결과 → skeleton 출력 JSON + 3D 리포트 (`_report.html`).

`measure_negotiate` 가 저장한 최종 경로를 skeleton `pipeline.run` 의 다중 배관 슬롯(planner)으로 꽂아,
skeleton 과 같은 검증·J 분해·3D 리포트(실제 굵기 관·서포트·위반 위치)를 만든다.

    PIPE_SUPPORT_STANDARD=... python3 v3/report_negotiate.py out/negotiate_opt_dense50.json.partial.jsonl \
        --scenario val_dense50_000 --manifest .../opt_stage1_valsets.json --outdir out/negotiate_viz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import standard_guard                                            # noqa: E402

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.multi import PlanResult, default_order                 # noqa: E402
from pipe_routing.pipeline import report_md, run, write_report_html     # noqa: E402
from pipe_routing.router_astar import RouteResult                       # noqa: E402

N_PIPES = {"val_dense30": 30, "val_dense50": 50}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--variant", default="C")
    ap.add_argument("--outdir", default="out/negotiate_viz")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    standard_guard.resolve(a.allow_public_standard, what="협상 3D 리포트")
    row = next(json.loads(l) for l in open(a.rows, encoding="utf-8") if json.loads(l)["scenario"] == a.scenario)
    v = row[a.variant]
    man = json.loads(pathlib.Path(a.manifest).read_text(encoding="utf-8"))
    ref = next(r for s in man["sets"].values() for r in s if pathlib.Path(r["file"]).stem == a.scenario)
    sc = generate(GeneratorConfig(seed=ref["seed"], n_pipes=N_PIPES[a.scenario.rsplit("_", 1)[0]]), a.scenario)
    wps, st1 = v["final_waypoints"], v["iter1_status"]

    def planner(sc_, router, time_limit):
        routes = {}
        for p in sc_.pipes:
            if p.id in wps:
                routes[p.id] = RouteResult(p.id, "ok", waypoints=[tuple(w) for w in wps[p.id]])
            else:
                s = st1.get(p.id, "unreachable")
                routes[p.id] = RouteResult(p.id, "timeout" if s == "budget_exhausted" else s)
        return PlanResult(routes=routes, order=[p.id for p in default_order(sc_)], planner=f"v3-negotiate-{a.variant}")

    out = run(sc, planner=planner, solo_ref=False)
    d = pathlib.Path(a.outdir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{a.scenario}_negotiate_{a.variant}"
    (d / f"{stem}_output.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (d / f"{stem}_report.md").write_text(report_md(out), encoding="utf-8")
    write_report_html(sc, out, d / f"{stem}_report.html")
    g = out["global"]
    print(json.dumps({k: g.get(k) for k in ("total_pipes", "routed_count", "success_count", "required_pipes",
                                           "required_passed", "full_success", "J_total")}, ensure_ascii=False))
    from collections import Counter
    print("실패 원인:", dict(Counter(c for r in out["routes"] if not r["success"] for c in r["fail_causes"])))
    print(d / f"{stem}_report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
