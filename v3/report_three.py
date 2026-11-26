"""한 맵의 세 해 (기준선 첫 해 · T exp_cap · T 상한 없음) 3D 리포트 + 통과가 다른 배관 표 (skeleton 사양 세션 요청 2026-11-26).

① 기준선 = skeleton `pipeline.run` (순차 첫 해, solo_ref 끔)
② T exp_cap = 상한이 걸린 반복의 스냅숏에서 확정 (D50) — `cap_from_snapshot.py` 와 같은 계산
③ T 상한 없음 = `<cap_out>` 의 final_waypoints
리포트는 skeleton viz (실제 굵기 관·위반 위치) 에 ② 와 ③ 사이 통과가 달라진 배관 강조 trace 를 더한다.
출력은 비공개 표 파생이라 레포에 커밋하지 않는다 (out/).

    PIPE_SUPPORT_STANDARD=... python3 v3/report_three.py --manifest <opt_stage1_valsets.json> --name val_dense50_001 \
        --cap-out out/cap_d50.json --ckpt-dir out/cap_d50.json.ckpt --cap-it 5 -o out/three_40263009
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from v3 import standard_guard                              # noqa: E402
from v3 import measure_negotiate as mn                     # noqa: E402
from v3.cap_from_snapshot import _load                     # noqa: E402

from pipe_routing.pipeline import j_breakdown, required_pipe_ids, run, write_report_html   # noqa: E402
from pipe_routing.verifier.core import PipeRoute, verify   # noqa: E402


def out_from_final(sc, final: dict, std, label: str) -> dict:
    """확정 경로 → write_report_html 이 읽는 최소 출력 dict (검증기 채점)."""
    rep = verify(sc, [PipeRoute(p.id, final.get(p.id) or []) for p in sc.pipes], std=std)
    routes = []
    for p in sc.pipes:
        v = rep["pipes"][p.id]
        wps = final.get(p.id) or []
        J = j_breakdown(p, wps, v.fittings, v.supports, sc) if v.routed else None
        routes.append({"pipe_id": p.id, "nominal_size": p.nominal_size, "success": v.success,
                       "waypoints": [[round(c, 3) for c in w] for w in wps],
                       "supports": v.supports, "support_gaps": v.support_gaps, "J": J,
                       "violations": [{"module": x.module, "pos": x.pos, "message": x.message, "other": x.other}
                                      for x in v.violations],
                       "router": {"status": "ok" if wps else "no_path"}})
    ok = [r for r in routes if r["success"]]
    req, _ = required_pipe_ids(sc)
    return {"scenario": label, "routes": routes,
            "global": {"success_count": len(ok), "total_pipes": len(routes),
                       "J_total": sum(r["J"]["total"] for r in ok),
                       "required_pipes": len(req), "required_passed": sum(r["pipe_id"] in req for r in ok),
                       "support_standard": rep["support_standard"]}}


def cause(r) -> str:
    if r["success"]:
        return "통과"
    if not r["waypoints"]:
        fc = r.get("fail_causes") or []
        cls = r.get("fail_class")
        s = ", ".join(fc) if fc else "경로 없음"
        return s + (f" ({cls})" if cls else "")
    mods = []
    for v in r["violations"]:
        m = v["module"] + (f" ← {v['other']}" if v.get("other") else "")
        if m not in mods:
            mods.append(m)
    return "위반: " + "; ".join(mods)


def write_html(sc, out, path, focus, title):
    """skeleton write_report_html 과 같은 그림 + 강조 배관 굵은 선·라벨."""
    import plotly.graph_objects as go
    from pipe_routing import viz
    calls = {}
    orig = viz.write_html

    def grab(fig, p, offline=False):
        calls["fig"] = fig
    viz.write_html = grab
    try:
        write_report_html(sc, out, path)
    finally:
        viz.write_html = orig
    fig = calls["fig"]
    for r in out["routes"]:
        if r["pipe_id"] not in focus:
            continue
        w = r["waypoints"]
        tag = "통과" if r["success"] else ("위반" if w else "경로 없음")
        if w:
            xs, ys, zs = zip(*w)
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=14, color="#ff00cc"),
                                       name=f"★ {r['pipe_id']} ({tag})", hovertext=cause(r), hoverinfo="text"))
        p = next(q for q in sc.pipes if q.id == r["pipe_id"])
        for t, lab in ((p.start, "start"), (p.end, "end")):
            fig.add_trace(go.Scatter3d(x=[t.pos[0]], y=[t.pos[1]], z=[t.pos[2]], mode="markers+text",
                                       marker=dict(size=7, color="#ff00cc", symbol="diamond"),
                                       text=[f"★{r['pipe_id']} {lab}"], showlegend=False,
                                       hovertext=f"{r['pipe_id']} {p.nominal_size} {lab} — {tag}", hoverinfo="text"))
    g = out["global"]
    fig.update_layout(title=dict(text=f"{title}<br>필수 통과 {g['required_passed']}/{g['required_pipes']} · "
                                      f"J_total {g['J_total']:,.1f} kg · ★ = 상한 때문에 빠진 배관 "
                                      f"{', '.join(sorted(focus))}"))
    orig(fig, path, offline=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dense50")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cap-out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cap-it", type=int, required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--baseline-only", action="store_true", help="기준선 첫 해만 돌려 저장 (메모리 때문에 따로)")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    std_info = standard_guard.resolve(a.allow_public_standard, what="세 해 3D 리포트")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")
    a.e1_sets = ""
    od = pathlib.Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    make = next(m for n, m, _ in mn.scenarios(a) if n == a.name)
    sc = make()
    bp = od / "baseline_output.json"
    if a.baseline_only:                     # 메모리 (셸 6 GB) — 기준선은 따로 프로세스에서 먼저
        t = time.perf_counter()
        import functools
        from pipe_routing.multi import sequential_ripup_planner
        # classify=False — D74 진단 재라우팅 (× 4 예산) 이 셸 6 GB 를 넘는다. 경로·통과는 같다 (D110 ②: 분류는 예산 밖)
        o1 = run(sc, solo_ref=False, planner=functools.partial(sequential_ripup_planner, classify=False))
        print(f"① 기준선 {time.perf_counter() - t:.0f} s · 필수 {o1['global']['required_passed']}/"
              f"{o1['global']['required_pipes']}", flush=True)
        bp.write_text(json.dumps(o1, ensure_ascii=False, default=float), encoding="utf-8")
        return 0

    # ③ 상한 없음
    rows = json.loads(pathlib.Path(a.cap_out).read_text(encoding="utf-8"))["rows"]
    unc = {k: [tuple(w) for w in v] for k, v in
           next(r for r in rows if r["scenario"] == a.name)["T"]["final_waypoints"].items()}
    o3 = out_from_final(sc, unc, std, f"{a.name} T 상한 없음")

    # ② exp_cap — 상한이 걸린 반복 스냅숏에서 확정
    st = _load(pathlib.Path(a.ckpt_dir) / f"{a.name}.T.it{a.cap_it:02d}.pkl")
    neg = mn.Negotiator(sc, mn.EXPAND_BUDGET)
    neg.toll = "self"
    neg.p = mn.variant_params("T")
    s = st["neg"]["state"]
    final, _ = mn.commit(neg, s["paths"], s["conf"], mn.Clock(st["neg"]["clk"]), list(st["neg"]["calls"]))
    del neg
    o2 = out_from_final(sc, final, std, f"{a.name} T exp_cap (반복 {a.cap_it})")
    print(f"② 필수 {o2['global']['required_passed']}/{o2['global']['required_pipes']} · "
          f"③ 필수 {o3['global']['required_passed']}/{o3['global']['required_pipes']}", flush=True)

    # ① 기준선 첫 해 (--baseline-only 로 먼저 만든 것)
    if not bp.exists():
        raise SystemExit("기준선 출력이 없다 — 먼저 --baseline-only 로 돌린다")
    o1 = json.loads(bp.read_text(encoding="utf-8"))
    if o1["global"]["support_standard"]["sha256"] != std_info["sha256"]:
        raise SystemExit("기준선 출력 서포트 표 해시 불일치 — 중단")

    outs = {"①": o1, "②": o2, "③": o3}
    by = {k: {r["pipe_id"]: r for r in o["routes"]} for k, o in outs.items()}
    focus = {p.id for p in sc.pipes if by["③"][p.id]["success"] and not by["②"][p.id]["success"]}
    for k, o in outs.items():
        o["global"].setdefault("required_passed", o["global"].get("required_passed"))
    titles = {"①": "① 기준선 첫 해", "②": f"② T exp_cap 180,481,864 (반복 {a.cap_it} 에서 멈춤)", "③": "③ T 상한 없음 (반복 15)"}
    fnames = {"①": "1_baseline_report.html", "②": "2_T_expcap_report.html", "③": "3_T_nocap_report.html"}
    for k in outs:
        write_html(sc, outs[k], od / fnames[k], focus, f"{a.name} (seed 40263009) — {titles[k]}")

    diff = [p for p in sc.pipes if len({by[k][p.id]["success"] for k in outs}) > 1]
    lines = [f"# {a.name} — 셋 중 통과가 다른 배관\n",
             "| | 필수 통과 | J_total (kg) |", "|---|---|---|"]
    for k in outs:
        g = outs[k]["global"]
        lines.append(f"| {titles[k]} | {g['required_passed']}/{g['required_pipes']} | {g['J_total']:,.1f} |")
    lines += ["", "★ = ② 에서 빠지고 ③ 에서 통과 (상한 때문에 빠진 배관)", "",
              "| 배관 | 구경 | 서비스 | ① 기준선 | ② exp_cap | ③ 상한 없음 |", "|---|---|---|---|---|---|"]
    for p in diff:
        cells = [("✅ " if by[k][p.id]["success"] else "❌ ") + cause(by[k][p.id]).replace("통과", "").strip()
                 for k in outs]
        lines.append(f"| {'★ ' if p.id in focus else ''}{p.id} | {p.nominal_size} | {p.service or '—'} | "
                     + " | ".join(c.replace("|", "/") or "✅" for c in cells) + " |")
    lines += ["", f"서포트 표: source = {std_info.get('source')} · sha256 {std_info.get('sha256', '')[:12]}…"]
    (od / "diff_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
