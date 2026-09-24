#!/usr/bin/env python3
"""끝난 런들의 체크포인트를 **현재 노트북 코드**로 다시 평가해 §7 기준으로 재판정한다.

지표 정의가 바뀐 뒤(2026-09-23: length_ratio → J_ratio·elbow_ratio, 3분위 계기판) 기존 런을
다시 돌리지 않고 같은 기준으로 비교하기 위한 것이다. 평가 함수는 노트북 셀 9 의 것을 그대로 쓴다.

사용: python3 regrade.py <런 디렉터리> [...] [--set k=v] [--out out.json]
      (캐시가 있는 디렉터리에서 실행할 것 — 시나리오·순위 라벨 캐시를 찾는다)
"""
import argparse, contextlib, io, json, os, sys
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_step1_ab import load_cells, NB, ARMS  # noqa: E402


def build_ns(override):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in load_cells(NB):
        if tag in ("3.", "4.", "5.", "6.", "7.", "7b.", "8."):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                for k, v in override.items():
                    setattr(ns["cfg"], k, v)
                ns["cfg"].pretrain_h = False
                ns["cfg"].auto_resume = False
        elif tag == "9.":
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src.split("_t0 = time.time()")[0], "<nb:9-defs>", "exec"), ns)
            break
    return ns


def se(v):
    v = [x for x in v if x == x]                       # NaN(해당 에피소드 없음) 은 뺀다
    return (stdev(v) / len(v) ** 0.5) if len(v) > 1 else float("nan")


def nmean(v):
    v = [x for x in v if x == x]
    return mean(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--ckpt", default="checkpoint_step1_final")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    override = dict(ARMS["L"])
    for kv in a.set:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except Exception: pass
        override[k] = v
    from sb3_contrib import MaskablePPO
    ns = build_ns(override)
    cfg = ns["cfg"]
    res = {}
    for d in a.dirs:
        f = os.path.join(d, a.ckpt + ".zip")
        if not os.path.exists(f):
            print("없음:", f); continue
        model = MaskablePPO.load(f, device="cpu", print_system_info=False)
        m = ns["evaluate_routing"](model, ns["EVAL_POOL"])
        m.update(ns["h_rank_metrics"](model.policy))
        m["timesteps"] = int(model.num_timesteps)
        res[os.path.basename(d.rstrip("/"))] = m
        print(f"{os.path.basename(d.rstrip('/'))} @ {m['timesteps']:,}: {ns['fmt_metrics'](m)}", flush=True)
    rs = list(res.values())
    if not rs:
        return
    keys = ["success_rate", "j_ratio", "deadlock_rate", "elbow_ratio", "j_ratio_all", "length_ratio",
            "succ_easy", "succ_mid", "succ_hard", "jr_easy", "jr_mid", "jr_hard", "hard_gap",
            "rank_top1", "rank_spearman",
            "steps_approach", "steps_approach_base", "reentries", "reentries_fail", "astar_overlap"]
    agg = {k: (nmean([r.get(k, float('nan')) for r in rs]), se([r.get(k, float('nan')) for r in rs])) for k in keys}
    print(f"\n=== §7 재판정 ({len(rs)}시드 평균 ± SE) ===")
    for nm, k, thr, op in (("success", "success_rate", cfg.grad_success, "≥"),
                           ("J_ratio (성공)", "j_ratio", cfg.grad_j_ratio, "≤"),
                           ("deadlock", "deadlock_rate", cfg.grad_deadlock, "≤"),
                           ("elbow_ratio (성공)", "elbow_ratio", cfg.grad_elbow_ratio, "≤")):
        mu, e = agg[k]
        ok = (mu >= thr) if op == "≥" else (mu <= thr)
        print(f"  {nm:<18} {mu:6.3f} ± {e:.3f}   기준 {op}{thr}   {'PASS' if ok else 'FAIL'}")
    print(f"  {'[목표선] J_ratio':<18} {agg['j_ratio'][0]:6.3f}           목표 ≤{cfg.target_j_ratio}")
    print(f"  {'[계기판] J_ratio 전체':<18} {agg['j_ratio_all'][0]:6.3f} ± {agg['j_ratio_all'][1]:.3f}")
    print(f"  {'[계기판] length_ratio':<18} {agg['length_ratio'][0]:6.3f}")
    print(f"  [계기판] 3분위 success  쉬움 {agg['succ_easy'][0]:.3f} / 보통 {agg['succ_mid'][0]:.3f} / "
          f"어려움 {agg['succ_hard'][0]:.3f}  (격차 {agg['hard_gap'][0]:+.3f}, 경보 ≥ {cfg.alarm_hard_gap})")
    print(f"  [계기판] 3분위 J_ratio  쉬움 {agg['jr_easy'][0]:.3f} / 보통 {agg['jr_mid'][0]:.3f} / "
          f"어려움 {agg['jr_hard'][0]:.3f}")
    print(f"  [계기판] 시드별 어려움 경보: " + " ".join(str(int(r['alarm_hard'])) for r in rs))
    print(f"  [계기판] top-1 {agg['rank_top1'][0]:.3f} · rho {agg['rank_spearman'][0]:+.3f}")
    print(f"  [방향] 진입 구간 스텝 {agg['steps_approach'][0]:.1f} ± {agg['steps_approach'][1]:.1f} "
          f"(A* {agg['steps_approach_base'][0]:.1f}) · 재진입 성공 {agg['reentries'][0]:.2f} / 실패 {agg['reentries_fail'][0]:.2f}")
    if a.out:
        json.dump(dict(per_run=res, agg=agg), open(a.out, "w"), ensure_ascii=False, indent=1, default=float)
        print("→", a.out)


if __name__ == "__main__":
    main()
