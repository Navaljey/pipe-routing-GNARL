"""상한 k 에서 끝난 T 의 결과 — 반복 k 스냅숏에서 확정·채점만 다시 한다 (설계안 §40.5).

반복 상한은 반복 경계에서만 보므로 "상한 = k 로 돌린 실행" 의 협상 상태 = "상한 없이 돌린 실행의 반복 k 스냅숏" 이다.
그래서 상한 후보마다 협상을 다시 돌리지 않고 스냅숏에서 확정 (D50 순서) + 검증기 1 회만 돈다.
스냅숏은 `V3_KEEP_ITERS=1 measure_negotiate.py --resume ...` 이 `<out>.ckpt/<이름>.T.itNN.pkl` 로 남긴다.

    PIPE_SUPPORT_STANDARD=... python3 v3/cap_from_snapshot.py --manifest <opt_stage1_valsets.json> --set dense50 \
        --ckpt-dir out/cap_d50.json.ckpt --out out/cap_d50_snap.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from v3 import guards, standard_guard                     # noqa: E402
from v3 import measure_negotiate as mn                    # noqa: E402


class _Unpickler(pickle.Unpickler):
    """스냅숏은 measure_negotiate.py 를 스크립트로 돌려 남겨서 클래스가 `__main__` 에 묶여 있다 → mn 에서 찾는다."""

    def find_class(self, module, name):
        if module == "__main__":
            return getattr(mn, name)
        return super().find_class(module, name)


def _load(path):
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dense50")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--variant", default="T")
    ap.add_argument("--out", required=True)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    std_info = standard_guard.resolve(a.allow_public_standard, what="상한 스냅숏 채점")
    from pipe_routing.support_std import load_standard
    std = load_standard()
    if std.info().get("sha256") != std_info.get("sha256"):
        raise SystemExit("서포트 표 해시 불일치 — 중단")
    a.e1_sets = ""
    out = []
    part = pathlib.Path(a.out + ".partial.jsonl")
    if part.exists():                                      # 끊겼다 다시 돌 때 — 이미 채점한 (시나리오, 반복) 은 건너뛴다
        out = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {(r["scenario"], r["it"]) for r in out}
    for name, make, ref in mn.scenarios(a):
        snaps = sorted(pathlib.Path(a.ckpt_dir).glob(f"{name}.{a.variant}.it*.pkl"))
        if not snaps:
            continue
        sc = make()
        for sp in snaps:
            st = _load(sp)
            if (name, st["neg"]["state"]["it"]) in done:
                continue
            neg =mn.Negotiator(sc, mn.EXPAND_BUDGET)
            neg.toll = "self"
            neg.p = mn.variant_params(a.variant)
            s = st["neg"]["state"]
            calls = list(st["neg"]["calls"])
            clk = mn.Clock(st["neg"]["clk"])
            t = time.perf_counter()
            final, n_commit = mn.commit(neg, s["paths"], s["conf"], clk, calls)
            t_commit = time.perf_counter() - t
            t = time.perf_counter()
            score = mn.score(sc, final, std)
            t_verify = time.perf_counter() - t
            row = {"scenario": name, "it": s["it"], "conflict_pairs": len(s["conf"]),
                   "cum_wall_at_it": s["trace"][-1]["cum_wall"], "expanded_cum_at_it": mn._expanded(st["neg"]["calls"]),
                   "commit_sec": round(t_commit, 2), "verify_sec": round(t_verify, 2), "commit_rerouted": n_commit,
                   "score": score}
            print(f"  {name} it {s['it']:2d}: 필수 {score['required_passed']}/{score['required_pipes']} · "
                  f"J {score['J_total']:.0f} · 벽시계(협상) {row['cum_wall_at_it']:.0f} s + 확정 {t_commit:.0f} s", flush=True)
            out.append(row)
            with open(a.out + ".partial.jsonl", "a", encoding="utf-8") as f:   # 2 시간 제한 대비 — 행마다 남긴다
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            del neg
    pathlib.Path(a.out).write_text(json.dumps({"support_standard": std_info, "rows": out}, ensure_ascii=False,
                                              indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
