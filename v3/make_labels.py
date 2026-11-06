"""1단계 라벨 생성 — 빈 맵 단독 A\\* 경로의 **상태별 남은 비용** (D86 ①, D-p).

무엇을 만드는가
---------------
학습 시드로 시나리오를 생성하고, 배관마다 **다른 배관·예약 없이** (빈 맵 단독)
skeleton 공식 컴파일 A\\* 를 돌려 최적 경로를 받는다. 그 경로의 **상태마다**
남은 비용을 세 갈래로 분해해 라벨로 쓴다.

    rem_pe[i]  = Σ (직관 + 엘보)      i → 목표
    rem_sup[i] = Σ 추정 서포트 (D57)   i → 목표
    rem_pen[i] = Σ 관통 피스 (D95)     i → 목표      (단일 블록은 전부 0)

    rem_total = rem_pe + rem_sup + rem_pen = 라우터 비용 기준 남은 최적값

**2머리로 쓴다 (D-p, 설계안 §14.1).** 서포트 항이 배관 항보다 늦게 배우는 것을
감시해야 하므로 (소유자 지시 2026-10-28 §5) 라벨 단계에서 이미 쪼개 둔다.
합계만 쓰면 서포트는 기울기의 8~9% 뿐이다.

    ĥ_pipe = h + softplus(머리₁)   목표 rem_pe    — h(D41) ≤ rem_pe 가 항상 성립
    ĥ_sup  =     softplus(머리₂)   목표 rem_sup   — h 는 서포트를 전혀 못 본다
    ĥ      = ĥ_pipe + ĥ_sup        FOCAL 순서는 합으로 (kg 단위 유지)

왜 잔차인가 — `h` 는 admissible 하한이므로 `rem_total ≥ rem_pe ≥ h` 가 항상 참이다.
잔차를 배우면 망이 이미 아는 부분을 다시 배우지 않고, **`h` 의 맹점(서포트·구배)에만**
용량을 쓴다. 그 맹점이 정확히 느슨함의 출처다 (설계안 §10.6, r = 0.657).

한계 — 솔직히 적는다
--------------------
라벨은 **최적 경로 위 상태뿐**이다. 포컬 탐색에서 망이 점수를 매기는 대상은 FOCAL 안
**전선 상태**이고 그중 대부분은 경로 밖이다. 경로 밖 상태의 `rem` 을 정확히 받으려면
상태마다 A\\* 를 다시 돌려야 해서(상태당 수 초) 비용이 맞지 않는다.
→ **회귀가 경로 밖에서도 순서를 맞추는지는 가정이 아니라 측정 대상이다.**
주 지표를 MAE 가 아니라 **FOCAL 안 순위 일치율**로 둔 이유다 (설계안 §14.3).

규약 (D-m · D86 · 계약 §d2)
---------------------------
* 서포트 표는 **비공개**여야 한다 — `standard_guard.resolve()` 가 막는다.
* 시드는 **학습 범위** 30000000~39999999 — `guards.assert_seed_range` 가 막는다.
* 매니페스트에 **skeleton 커밋 · 표 sha256 · 격자 변형**을 기록한다.
* 라벨은 `labels/` 에 둔다. **공개 레포에 커밋하지 않는다** (D78 — 서포트 중량이 들어 있다).

    PIPE_SUPPORT_STANDARD=/home/claude/pipe-support-standard \\
      python3 v3/make_labels.py --seed0 30000000 --n 20 --out labels/stage1_pilot
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.generator import GeneratorConfig, generate           # noqa: E402
from pipe_routing.layered import (EXPAND_BUDGET, G0, LayeredGraph,      # noqa: E402
                                  _route, clear_cache, heuristic_factory)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from v3 import guards, standard_guard                                   # noqa: E402

#: E1 기본 세트와 **같은 생성 설정**으로 둔다 — 학습 분포가 평가 분포와 어긋나면
#: 1단계 측정이 분포 차이를 재게 된다. 평가 시나리오의 meta.config 와 맞춘 값이다
#: (obstacle_fill 0.12 · n_pipes 10 · gravity_ratio 0.3 · stair_rule·d94_rule 없음).
GEN_KW = dict(n_pipes=10)


def skeleton_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(SK), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def label_route(sc, pipe, budget: int) -> dict:
    """배관 1개 — 빈 맵 단독 최적 경로와 상태별 남은 비용 분해."""
    t0 = time.perf_counter()
    g = LayeredGraph(sc, pipe, (), local_lines=False, support_cost=True, reserved=(), grid=G0)
    h = heuristic_factory(g, pipe)
    build_sec = time.perf_counter() - t0
    r = _route(g, pipe, 1e9, budget)        # 결정적 예산 (D72·D97). 벽시계 상한은 쓰지 않는다
    row = {"pipe_id": pipe.id, "size": pipe.nominal_size, "gravity": bool(pipe.gravity_pipe),
           "status": r.status, "expanded": r.expanded, "generated": r.generated,
           "build_sec": round(build_sec, 3), "search_sec": round(r.search_sec, 3)}
    if r.status != "ok":
        return row

    st = r.states
    n = len(st)
    # 뒤에서부터 누적. cost = 직관 + 엘보 + 추정 서포트 + 관통 (layered.cost, D57·D95)
    rem_pe = [0.0] * n
    rem_sup = [0.0] * n
    rem_pen = [0.0] * n
    for i in range(n - 2, -1, -1):
        a, b = st[i], st[i + 1]
        tot = g.cost(a, b, pipe)
        sup = g.support_cost(a, b)
        pen = g.penetration_cost(a, b)
        rem_sup[i] = rem_sup[i + 1] + sup
        rem_pen[i] = rem_pen[i + 1] + pen
        rem_pe[i] = rem_pe[i + 1] + (tot - sup - pen)

    hv = [h(s) for s in st]
    # admissible 검사 — h ≤ rem_pe 가 깨지면 라벨이 아니라 skeleton 쪽 문제다. 조용히 넘기지 않는다
    viol = [i for i in range(n) if hv[i] > rem_pe[i] + 1e-6]
    row["h_violations"] = len(viol)

    row.update(
        J=r.J, J_pipe=r.J_pipe, J_elbow=r.J_elbow, J_support_est=r.J_support_est,
        J_penetration=r.J_penetration, length_mm=r.length_mm, bends=r.bends,
        n_states=n,
        # 상태 — skeleton `space.State` 의 모든 성분을 그대로 (특징은 나중에 다시 뽑는다)
        node=[int(s.node) for s in st],
        pos=[[round(c, 3) for c in g.position(s.node)] for s in st],
        dir=[int(s.dir) for s in st],
        run=[round(float(s.run), 6) for s in st],
        bend=[int(s.bend) for s in st],
        ends=[int(s.ends) for s in st],
        hz=[round(float(s.hz), 6) for s in st],
        su=[round(float(s.su), 6) for s in st],
        h=[round(v, 6) for v in hv],
        rem_pe=[round(v, 6) for v in rem_pe],
        rem_sup=[round(v, 6) for v in rem_sup],
        rem_pen=[round(v, 6) for v in rem_pen],
    )
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="1단계 빈 맵 단독 라벨 생성")
    ap.add_argument("--seed0", type=int, default=30_000_000, help="첫 시드 (학습 범위)")
    ap.add_argument("--n", type=int, default=20, help="시나리오 수")
    ap.add_argument("--stride", type=int, default=1, help="시드 간격 (샤딩용)")
    ap.add_argument("--out", default="labels/stage1", help="출력 접두 (.jsonl · .manifest.json)")
    ap.add_argument("--budget", type=int, default=EXPAND_BUDGET, help=f"확장 예산 (기본 {EXPAND_BUDGET} = 2²²)")
    ap.add_argument("--purpose", default="train", choices=["train", "valid"])
    ap.add_argument("--throwaway", action="store_true", help="폐기용 표시 (학습 루프가 거른다)")
    standard_guard.add_argument(ap)
    a = ap.parse_args(argv)

    seeds = [a.seed0 + i * a.stride for i in range(a.n)]
    guards.assert_seed_range(seeds, a.purpose)                      # ★ 평가 시드 오염 차단
    std = standard_guard.resolve(a.allow_public_standard, what="라벨 생성")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = out.with_suffix(".jsonl")
    attempt = out.with_suffix(".attempt")

    done: set[tuple[str, str]] = set()
    if stream.exists():
        for line in stream.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["scenario"], r["pipe_id"]))
    # OOM 킬로 죽은 라우트는 흔적만 남는다 — 실패로 기록하고 넘어간다 (h_slack 에서 쓴 장치)
    if attempt.exists():
        key = tuple(attempt.read_text(encoding="utf-8").strip().split("\t"))
        if len(key) == 2 and key not in done:
            row = {"scenario": key[0], "pipe_id": key[1], "status": "oom_killed"}
            with stream.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add(key)
        attempt.unlink(missing_ok=True)

    manifest = guards.make_manifest(
        std_sha256_value=std.get("sha256") or "PUBLIC_TEMPORARY",
        skeleton_commit=skeleton_commit(), grid_variant=G0.name,
        purpose=a.purpose, throwaway=a.throwaway,
        support_standard=std, budget=a.budget, seeds=[seeds[0], seeds[-1]], n_scenarios=len(seeds),
        stride=a.stride, generator_kw=GEN_KW, label="stage1_solo_empty_map",
        note="빈 맵 단독 A* 최적 경로의 상태별 남은 비용 분해 (rem_pe · rem_sup · rem_pen)",
    )
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    t_all = time.perf_counter()
    n_ok = n_states = 0
    for k, seed in enumerate(seeds):
        name = f"train_{seed}"
        # skeleton 고정층 LRU(`_FIXED`)는 단일 블록 층을 쌓아 두기만 한다 — 시나리오를
        # 넘어갈 때 비우지 않으면 RSS 가 계속 올라가 OOM 으로 죽는다 (샤드 A 실측).
        # 시나리오 안에서는 (구경, 중력관) 별로 캐시가 유효하므로 경계에서만 비운다
        clear_cache()
        try:
            sc = generate(GeneratorConfig(seed=seed, **GEN_KW), name)
        except Exception as e:                                   # 생성 실패 시드는 건너뛴다
            print(f"  [{k + 1}/{len(seeds)}] {name}: 생성 실패 {e}", flush=True)
            continue
        for pipe in sc.pipes:
            if (name, pipe.id) in done:
                continue
            attempt.write_text(f"{name}\t{pipe.id}", encoding="utf-8")
            row = label_route(sc, pipe, a.budget)
            attempt.unlink(missing_ok=True)
            row = {"scenario": name, "seed": seed, **row}
            with stream.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row["status"] == "ok":
                n_ok += 1
                n_states += row["n_states"]
        print(f"  [{k + 1}/{len(seeds)}] {name}: ok {n_ok} · 상태 {n_states} · "
              f"{round(time.perf_counter() - t_all)}s", flush=True)

    print(f"\n라벨 {stream} — ok 라우트 {n_ok} · 라벨 상태 {n_states} · "
          f"{round(time.perf_counter() - t_all)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
