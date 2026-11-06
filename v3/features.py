"""신경망 ① 입력 특징 — 상태 1개 → 벡터 (D-p 확정, 2026-10-28).

설계 제약 두 가지가 모든 선택을 정한다.

1. **탐색 중에 상태마다 계산된다.** 기준선 확장 중앙이 272,922 이고 판정이 벽시계이므로
   (D98) 상태당 비용이 마이크로초 단위여야 한다. 그래서 특징은 **저장된 상태 성분 +
   라우터가 이미 캐시한 배열의 O(1) 색인**만으로 만든다. `support_hint.form_at` 처럼
   ms 단위 호출은 **금지**다 — 272,922 × 1.35 ms ≈ 368초 = 기준선의 37배 (설계안 §14.2).
2. **크기 비의존.** 2단계에서 맵이 커지므로 절대 좌표가 아니라 목표 상대 좌표와
   대각 정규화를 쓴다 (D-r 권고).

서포트 맥락이 **선택이 아니라 요건**인 이유 (설계안 §10): `h`(D41) 는 서포트 비용을
전혀 못 본다. 그 맹점이 느슨함의 출처고 (서포트 비중 ↔ 느슨함 r = 0.657),
**FOCAL 안 순서로 그 맹점을 메우는 것**이 이 망의 일이다. 망이 `h` 와 같은 정보만
보면 순서를 바꿀 재료가 없다.

장애물 속성 2칸은 **미리 비워 둔다** (S2 예고, 소유자 권고 2026-10-28 §6).
나중에 채널을 더하면 freeze + zero-padding 이 충돌한다 (D-n §2.1) — 0 으로 둔
가중치는 freeze 하면 학습되지 않아 새 채널이 죽은 입력이 된다.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np

SK = pathlib.Path(os.environ.get("SKELETON_ROOT", "/home/claude/navaljey/pipe-routing-skeleton"))
if str(SK) not in sys.path:
    sys.path.insert(0, str(SK))

from pipe_routing.constants import FITTINGS, PIPE_SPECS                 # noqa: E402
from pipe_routing.space import DIRS                                     # noqa: E402

N_DIRS = len(DIRS)          # 18 — escape graph (D41)
N_BEND = 4                  # 0 / 45 / 90 / 135
BEND_IDX = {0: 0, 45: 1, 90: 2, 135: 3}

#: 정규화 상수. 2단계에서 맵이 커져도 **같은 값을 쓴다** — 특징이 맵 크기에 따라
#: 바뀌면 1단계 망을 그대로 이어받을 수 없다 (D-r).
DIAG_MM = math.dist((0, 0, 0), (40000, 40000, 10000))   # 단일 블록 대각 ≈ 57,446 mm
KG_PER_M_MAX = PIPE_SPECS["500A"].kg_per_m               # 185.0
R_MAX = PIPE_SPECS["500A"].od / 2                        # 254 mm

NAMES: list[str] = (
    ["goal_dx", "goal_dy", "goal_dz", "goal_dist", "pos_x", "pos_y", "pos_z", "h_norm"]
    + [f"dir_{i}" for i in range(N_DIRS)]
    + [f"bend_{a}" for a in (0, 45, 90, 135)]
    + ["run_ratio"] + [f"ends_{i}" for i in range(4)]
    + ["dir_dot_enddir"] + [f"enddir_{i}" for i in range(N_DIRS)]
    + ["su_used", "su_slack", "edge_support_cost", "node_support_frac", "out_of_table"]
    + ["is_gravity", "hz_ratio"]
    + ["kg_per_m", "elbow_in_m", "radius", "turn_need_90", "bend_u_90"]
    + ["obs_supportable", "obs_penetrable"]          # S2 예약 — 지금은 0
)
DIM = len(NAMES)


class FeatureMaker:
    """배관 1개용. 그래프 배열을 잡아 두고 상태마다 O(1) 로 벡터를 만든다."""

    def __init__(self, g, pipe):
        self.g = g
        self.pipe = pipe
        spec = PIPE_SPECS[pipe.nominal_size]
        fit = FITTINGS[pipe.nominal_size]
        self.goal = np.asarray(pipe.end.pos, float)
        self.end_dir = tuple(pipe.end.dir)
        self.end_dir_idx = DIRS.index(self.end_dir) if self.end_dir in DIRS else -1
        self.kg_per_m = spec.kg_per_m
        self.e90 = fit.elbow90
        self.r = spec.od / 2
        self.is_grav = bool(pipe.gravity_pipe)
        self.lmax = float(getattr(g, "lmax", 0.0)) or 1.0
        self.j_scale = DIAG_MM / 1000 * self.kg_per_m        # 이 배관의 J 척도 (G1-Q2 와 같은 발상)
        self.turn90 = float(g.run_cap.get(90, g.L))
        # `State.su` 는 **이미 정규화된 값**이다 — support_step 이 `u + length/smax` 로 누적하고
        # `u ≤ 1` 이 D80 ⑨ 제약이다 (support_router.support_step). mm 로 나누면 안 된다.
        self.bend_u90 = float(np.asarray(g.bend_u, float)[2]) if getattr(g, "bend_u", None) is not None else 0.0
        # 노드별 "서포트 가능 출구" 비율 — spack 0 = 제약 없음, 그 밖 = 구간 요약 (D80 ⑨)
        self.scost = g.scost
        self.step = g.step

    def one(self, s, h_val: float) -> np.ndarray:
        g = self.g
        v = np.zeros(DIM, dtype=np.float32)
        p = np.asarray(g.position(s.node), float)
        d = self.goal - p
        dist = float(np.linalg.norm(d))
        v[0:3] = d / DIAG_MM
        v[3] = dist / DIAG_MM
        v[4:7] = p / DIAG_MM
        v[7] = h_val / self.j_scale
        i = 8
        v[i + int(s.dir)] = 1.0
        i += N_DIRS
        v[i + BEND_IDX[int(s.bend)]] = 1.0
        i += N_BEND
        v[i] = min(float(s.run) / self.turn90, 2.0)
        i += 1
        for b in range(4):
            v[i + b] = (int(s.ends) >> b) & 1
        i += 4
        dv = DIRS[int(s.dir)]
        nd = math.sqrt(sum(c * c for c in dv))
        v[i] = sum(a * b for a, b in zip(dv, self.end_dir)) / (nd * math.sqrt(
            sum(c * c for c in self.end_dir))) if self.end_dir != (0, 0, 0) else 0.0
        i += 1
        if self.end_dir_idx >= 0:
            v[i + self.end_dir_idx] = 1.0
        i += N_DIRS

        # ── 서포트 맥락 (★ D-p 요건) ──────────────────────────────────────────
        su = min(float(s.su), 1.0)
        v[i] = su                                      # 최대 간격 중 쓴 비율 = "지지가 급한가" (D80 ⑨: u ≤ 1)
        v[i + 1] = 1.0 - su                            # 남은 여유 — 꺾으면 bend_u 만큼 더 먹는다
        row = self.scost[s.node]
        ec = float(row[int(s.dir)])                    # 들어온 방향의 엣지 서포트 비용
        v[i + 2] = ec / self.e90 if self.e90 else 0.0  # 엘보 90 중량 단위 — 구경 비의존
        out_ok = self.step[s.node] >= 0
        n_out = int(out_ok.sum())
        v[i + 3] = (float((row[out_ok] > 0).sum()) / n_out) if n_out else 0.0
        # 표 범위 밖 표시 (D109): 간격을 많이 썼는데 서포트 비용이 0 이면 지지가 안 걸린 구간이다
        v[i + 4] = 1.0 if (ec == 0.0 and su > 0.5) else 0.0
        i += 5

        # ── 중력관 (★ D-p 요건) ──────────────────────────────────────────────
        v[i] = 1.0 if self.is_grav else 0.0
        v[i + 1] = min(float(s.hz) / self.lmax, 2.0) if self.is_grav else 0.0
        i += 2

        # ── 배관 규격 ───────────────────────────────────────────────────────
        v[i] = self.kg_per_m / KG_PER_M_MAX
        v[i + 1] = self.e90 / self.kg_per_m            # 엘보 1개 = 직관 몇 m (§3 "엘보가 지배적")
        v[i + 2] = self.r / R_MAX
        v[i + 3] = self.turn90 / DIAG_MM
        v[i + 4] = self.bend_u90          # 90° 꺾을 때 먹는 서포트 간격 비율 (M29 가 걸리는 자리)
        i += 5
        # i, i+1 = 장애물 속성 예약 (S2) — 0 으로 둔다
        return v


def from_label_row(g, pipe, row) -> np.ndarray:
    """라벨 한 줄(ok 라우트)의 모든 상태 → (n_states, DIM)."""
    from pipe_routing.space import State
    fm = FeatureMaker(g, pipe)
    out = np.zeros((row["n_states"], DIM), dtype=np.float32)
    for k in range(row["n_states"]):
        s = State(row["node"][k], row["dir"][k], row["run"][k], row["bend"][k],
                  row["ends"][k], row["hz"][k], row["su"][k])
        out[k] = fm.one(s, row["h"][k])
    return out
