"""C안 학습 라벨 — 빈 맵 단독 컴파일 A\\* 최적 경로 위 **노드**의 h*/rg (설계안 §33, 2026-11-19).

`make_labels.label_route` 와 같은 경로·같은 남은 비용 분해를 쓰고, 같은 프로세스에서 그래프를 잡은 채로
`rg` (§30.2 relaxed_dist) 와 **노드 특징**(`v3.node_table.node_features`) 을 붙여 npz 로 바로 쓴다
(그래프를 두 번 만들지 않는다 — 배관당 build 약 5 s).

한 노드에 경로 상태가 둘 이상이면 h* 의 **최솟값**을 라벨로 쓴다 (노드 표는 그 노드의 모든 상태에
같은 값을 주므로 admissible 쪽으로).

    PIPE_SUPPORT_STANDARD=... python3 v3/make_node_labels.py --seed0 30000000 --n 30 --stride 2 --out labels/node_sA
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                    # noqa: E402
from v3.make_labels import GEN_KW, skeleton_commit                       # noqa: E402
from v3.measure_controls import relaxed_dist                             # noqa: E402
from v3.node_table import NAMES, node_features                           # noqa: E402

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import EXPAND_BUDGET, G0, LayeredGraph, _route, clear_cache  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, default=30_000_000)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET)
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    seeds = [a.seed0 + i * a.stride for i in range(a.n)]
    guards.assert_seed_range(seeds, "train")
    std = standard_guard.resolve(a.allow_public_standard, what="C안 라벨 생성")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".routes.jsonl")
    X, H, Y, SZ, RT = [], [], [], [], []
    rid = 0
    t_all = time.perf_counter()
    for k, seed in enumerate(seeds):
        name = f"train_{seed}"
        clear_cache()
        sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        for pipe in sc.pipes:
            g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
            r = _route(g, pipe, 1e9, a.budget)
            row = {"scenario": name, "pipe_id": pipe.id, "size": pipe.nominal_size, "status": r.status,
                   "expanded": r.expanded}
            if r.status == "ok":
                st = r.states
                n = len(st)
                rem = [0.0] * n
                for i in range(n - 2, -1, -1):
                    rem[i] = rem[i + 1] + g.cost(st[i], st[i + 1], pipe)
                rg, _ = relaxed_dist(g, pipe)
                best: dict[int, float] = {}
                for s, v in zip(st, rem):
                    nd = int(s.node)
                    if v > 1e-9 and np.isfinite(rg[nd]) and rg[nd] > 1e-9:
                        best[nd] = min(best.get(nd, np.inf), v)
                idx = np.asarray(sorted(best), dtype=np.int64)
                if len(idx):
                    hs = np.asarray([best[i] for i in idx])
                    viol = int((rg[idx] > hs + 1e-3).sum())
                    row.update(n_nodes=int(len(idx)), rg_viol=viol, J=r.J,
                               slack_start=float(r.J / rg[int(st[0].node)]))
                    X.append(node_features(g, pipe, rg, idx))
                    H.append(rg[idx])
                    Y.append(hs)
                    SZ += [pipe.nominal_size] * len(idx)
                    RT += [rid] * len(idx)
                    rid += 1
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            del g
        if X:
            Xa = np.concatenate(X)
            np.savez(out.with_suffix(".npz"), X=Xa, h=np.concatenate(H), y_pe=np.concatenate(Y),
                     y_sup=np.zeros(len(Xa)), y_pen=np.zeros(len(Xa)), size=np.asarray(SZ),
                     route=np.asarray(RT), names=np.asarray(NAMES), support_standard=np.asarray([std.get("sha256")]),
                     skeleton_commit=np.asarray([skeleton_commit()]))
        print(f"  [{k + 1}/{len(seeds)}] {name}: routes {rid} · nodes {sum(len(x) for x in X)} · "
              f"{round(time.perf_counter() - t_all)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
