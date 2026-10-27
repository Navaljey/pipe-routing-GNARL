"""라벨(.jsonl) → 학습 배열(.npz). 특징은 **다시 뽑을 수 있게** 분리해 둔다.

왜 두 단계인가 — 라벨(A\\* 가 말한 것: 상태열과 남은 비용 분해)은 **정본**이고 바뀌지
않는다. 특징 설계는 **바뀐다** (D-p 채널 추가, S2 장애물 속성). 한 파일에 섞으면
특징을 고칠 때마다 A\\* 를 다시 돌려야 한다 — 라벨 1,500 라우트가 약 2시간이다.

시나리오는 **시드에서 다시 생성**한다 (생성기가 결정적이고 평가 세트를 비트 동일하게
재현하는 것을 확인했다). 그래프도 다시 짓는다 — 그래서 특징이 탐색 중 값과 같다.

    PIPE_SUPPORT_STANDARD=/home/claude/pipe-support-standard \\
      python3 v3/make_features.py --labels labels/stage1_sA.jsonl labels/stage1_sB.jsonl \\
        --out labels/stage1_feat.npz
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.generator import GeneratorConfig, generate            # noqa: E402
from pipe_routing.layered import G0, LayeredGraph, clear_cache          # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import features, guards, standard_guard                         # noqa: E402

GEN_KW = dict(n_pipes=10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="라벨 → 특징 배열")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="라우트 수 상한 (시험용)")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)
    std = standard_guard.resolve(a.allow_public_standard, what="특징 추출")

    # ★ (시나리오, 배관) 로 **중복을 제거한다.** 라벨 파일은 재시작하며 덧붙이므로 같은
    # 라우트가 두 번 들어갈 수 있다 (실측: 샤드가 겹쳐 돌아 2건). 중복을 그냥 쓰면 그
    # 상태들이 학습에서 두 배 가중되고 **라우트 단위 분할도 깨진다** (같은 경로가 학습·
    # 검증 양쪽에 들어간다). 조용히 지나가면 안 되는 종류의 사고다.
    by_key: dict[tuple, dict] = {}
    n_lines = 0
    for path in a.labels:
        for line in pathlib.Path(path).open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_lines += 1
            if r.get("status") == "ok":
                by_key[(r["scenario"], r["pipe_id"])] = r        # 마지막 것을 쓴다
    rows = list(by_key.values())
    seeds = {r["seed"] for r in rows}
    print(f"라벨 줄 {n_lines} → ok 고유 라우트 {len(rows)}")
    guards.assert_seed_range(sorted(seeds), "train")     # 라벨 파일이 섞였는지 여기서도 본다
    rows.sort(key=lambda r: (r["scenario"], r["pipe_id"]))
    if a.limit:
        rows = rows[:a.limit]
    print(f"ok 라우트 {len(rows)} · 시나리오 {len(seeds)} · 상태 {sum(r['n_states'] for r in rows)}")

    X: list[np.ndarray] = []
    y_pe: list[np.ndarray] = []
    y_sup: list[np.ndarray] = []
    y_pen: list[np.ndarray] = []
    hv: list[np.ndarray] = []
    meta: list[tuple] = []        # (라우트 인덱스, 시드, 구경, 중력관) — 밴드별 보고용 (D-v)
    t0 = time.perf_counter()
    cur_seed = None
    sc = None
    for ri, r in enumerate(rows):
        if r["seed"] != cur_seed:
            # skeleton 고정층 LRU(`_FIXED`)는 단일 블록 층을 쌓아 두기만 한다 — 시나리오를
            # 넘어갈 때 비우지 않으면 수백 라우트에서 RSS 가 계속 올라가 OOM 이다 (실측:
            # 225 라우트 중 약 60 라우트에 3.4 GB). 시나리오가 바뀌면 캐시는 어차피 못 쓴다
            clear_cache()
            sc = generate(GeneratorConfig(seed=r["seed"], **GEN_KW), r["scenario"])
            cur_seed = r["seed"]
        pipe = next(p for p in sc.pipes if p.id == r["pipe_id"])
        g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
        X.append(features.from_label_row(g, pipe, r))
        y_pe.append(np.asarray(r["rem_pe"], dtype=np.float32))
        y_sup.append(np.asarray(r["rem_sup"], dtype=np.float32))
        y_pen.append(np.asarray(r["rem_pen"], dtype=np.float32))
        hv.append(np.asarray(r["h"], dtype=np.float32))
        meta += [(ri, r["seed"], r["size"], int(r["gravity"]))] * r["n_states"]
        if (ri + 1) % 25 == 0:
            print(f"  {ri + 1}/{len(rows)} · {round(time.perf_counter() - t0)}s", flush=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.concatenate(X), y_pe=np.concatenate(y_pe), y_sup=np.concatenate(y_sup),
        y_pen=np.concatenate(y_pen), h=np.concatenate(hv),
        route=np.asarray([m[0] for m in meta], dtype=np.int32),
        seed=np.asarray([m[1] for m in meta], dtype=np.int64),
        size=np.asarray([m[2] for m in meta]),
        gravity=np.asarray([m[3] for m in meta], dtype=np.int8),
        names=np.asarray(features.NAMES),
        support_standard=np.asarray([json.dumps(std, ensure_ascii=False)]),
    )
    print(f"\n{out} — 상태 {sum(len(x) for x in X)} × {features.DIM} 특징 · "
          f"{round(time.perf_counter() - t0)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
