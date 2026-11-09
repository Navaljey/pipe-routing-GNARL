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
    # ── 장애물 기하 (2026-11-09 소유자 지시) ───────────────────────────────
    # **왜 뒤에 붙이나:** 앞 68칸의 색인을 보존해야 기존 라벨·체크포인트·보고서의
    # 채널 이름이 그대로 읽힌다. 채널을 더할 수 있는 창은 **2단계가 끝날 때 닫힌다**
    # — freeze 는 계약 §(e2) 의 2→3 전환에서 한 번뿐이다 (D-n §2.1). 1단계와 2단계
    # 사이에는 동결이 없다 (2단계는 1단계 망을 계승해 계속 학습한다, §7.1).
    + [f"elen_{i}" for i in range(N_DIRS)] + ["n_out_ratio", "clearance_r"]
    # ── 관통 창 (2026-11-09 소유자 지시 "서포트 설치가능·관통 가능 정보도") ─────
    # 블록 쌍에서만 값이 생긴다. 단일 블록은 전부 0 (= 죽은 채널) — 의도된 것이다.
    + ["win_active", "win_dx", "win_crossed", "win_dy", "win_dz", "win_aimed"]
    # ── 장애물 **속성** (2026-11-09 소유자 지시) ───────────────────────────
    # 기하만으로는 탱크·관통 가능한 판·지지 가능한 구조재가 **똑같이 "막힘"** 으로
    # 보인다. 셋은 라우터가 할 수 있는 일이 전혀 다르다 (돌아가야 함 / 뚫고 가면 최단 /
    # 거기 매달면 u 가 0 으로 돌아감). D93 이 스키마에 둔 속성이 그것이다.
    + ["blk_pen_frac", "blk_sup_frac", "fwd_pen", "fwd_sup", "pen_near_r", "sup_near_r"]
)
DIM = len(NAMES)
OBS_SLICE = slice(68, 88)       #: 장애물 기하 채널 — ablation·대조군에서 이 구간만 조작한다
WIN_SLICE = slice(88, 94)       #: 관통 창 채널 — 같은 방식으로 대조군을 만든다
OBS_ATTR_SLICE = slice(94, 100)  #: 장애물 속성 채널. **66:68 의 2칸 전역 플래그를 대체한다**
#: — 그 2칸은 색인 보존을 위해 0 으로 남겨 둔다 (앞 68칸을 읽는 체크포인트 호환, §26.2)


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
        # ── 장애물 기하용 캐시 (2026-11-09). 전부 라우터가 이미 만들어 둔 배열이고
        # 행 3개 읽기 실측 **0.271 µs** 다 (특징 전체 15.6 µs 대비 무시 가능).
        # `elen == 0` ⟺ `step < 0` 이 **정확히 일치**함을 확인했으므로 (차단 칸 수 0)
        # 차단 마스크를 따로 두지 않는다 — `elen` 한 묶음이 차단과 자유거리를 다 담는다.
        #
        # ★ **그래프당 1회 사전계산.** `elen`·`clearance` 는 그래프마다 고정이므로 상태마다
        # 다시 계산할 이유가 없다. 상태마다 `np.log1p` 를 부르면 특징 비용이 15.6 → 19.97 µs
        # (+28%) 가 되고, 판정이 벽시계라 그만큼이 학습 효과를 깎아먹는다 (D98·설계안 §15).
        # 빌드 비용은 2.7M 원소 log1p ≈ 20 ms 로 `build_sec`(약 5.4 s) 대비 무시 가능하다.
        inv_log_diag = 1.0 / math.log1p(DIAG_MM)
        # 로그 척도 — 실측 분포가 10 ~ 52,736 mm (중앙 440 · p90 4,327) 로 꼬리가 두껍다
        self.elen_log = (np.log1p(g.elen) * inv_log_diag).astype(np.float32)
        self.n_out_ratio = ((g.elen > 0).sum(1) / N_DIRS).astype(np.float32)
        # 여유는 **구경 상대**로 본다 ("반지름 몇 개만큼 공간이 있나") — 크기 비의존 (D-r).
        # inf = 근처에 장애물 없음 → 1.0 (`np.minimum` 이 inf 를 그대로 잘라준다)
        clr = np.asarray(g.clearance, dtype=float)
        self.clr_r = np.minimum(np.nan_to_num(clr, nan=0.0, posinf=1e18)
                                / (20.0 * self.r), 1.0).astype(np.float32)
        # ── 관통 창 (2026-11-09) ────────────────────────────────────────────
        # **왜 비용이 아니라 기하인가.** 관통 피스 무게(D95)는 `kg/m × 판 두께` 이고
        # 실측 200A·20 mm 에서 **0.85 kg** 이다 — J 가 수천 kg 인 라우트에서 무시값이라
        # 비용 채널로는 쓸모가 없다. 값이 있는 것은 **기하 제약**이다:
        #   ① 관통 배관은 공유 격벽을 **창 안에서 정확히 1회, 수직으로** 지나야 한다 (D94)
        #   ② **관통부는 서포트 고정점**이라 `u` 가 0 으로 돌아간다 (D94, layered.py:352)
        # ② 의 *결과*는 이미 `su` 로 보인다. 못 보는 것은 **예고**다 — 그리고 `h`(D41)는
        # 유클리드 거리 + 최소 꺾임이라 **창의 존재를 전혀 모른다.** 창이 직선에서 멀면
        # `h` 가 크게 느슨해지고, 그 느슨함이 곧 망의 자리다 (설계안 §10 과 같은 논리).
        self.win = None
        win = getattr(pipe, "crossing_window", None)
        sc = getattr(g, "_sc", None)
        if win is not None and sc is not None and getattr(sc, "block_pair", None):
            from pipe_routing import blockpair as _bp           # 읽기만 (계약 §(b) 불변)
            y0, y1, z0, z1 = (float(v) for v in win)
            self.win = (_bp.center(sc), (y0 + y1) / 2, (z0 + z1) / 2,
                        max((y1 - y0) / 2, 1.0), max((z1 - z0) / 2, 1.0))
        self._init_obs_attr(sc)

    def _init_obs_attr(self, sc) -> None:
        """장애물 속성 사전계산 — 그래프당 1회.

        **속성이 붙은 장애물만 돈다.** E1 은 로더가 속성 있는 장애물을 거부하므로
        (`scenario.py:335`, M27) 목록이 비고, 이 함수는 아무것도 계산하지 않는다 →
        상태당 비용 0. 값이 생기는 것은 S2 / 평가셋 E2 (= V3 2단계) 부터다.

        방향 d 를 막은 것 = 그 방향으로 **`step` 사슬을 끝까지 따라간** 마지막 노드에 닿은 장애물.

        ⚠️ `elen[n, d]` 는 자유 직진 길이가 **아니다** — `step[n, d]` 노드까지의 **엣지 길이**다
        (`layered.py:792`). 처음 구현은 `pos + d·elen` 을 끝점으로 써서 끝점이 장애물 면이 아니라
        그냥 다음 격자선에 떨어졌고, 속성이 거의 안 걸렸다 (진행 방향 적중 0).
        사슬의 마지막 노드는 축 방향이면 **팽창 장애물 면(= 격자선, D38)** 위에 있으므로
        박스를 r + 20 mm 팽창한 범위 안에 든다.
        `추정:` 45° 방향은 엣지가 좌표 우연 일치로만 생겨(D38·D42) 장애물 없이도 사슬이 끊길 수
        있다 — 그때는 속성 박스에 안 닿아 0 이 된다 (거짓 양성 없음, 거짓 음성만).
        """
        self.obs_attr = None
        obs = [o for o in getattr(sc, "obstacles", ()) or ()
               if getattr(o, "supportable", False) or getattr(o, "penetrable", False)]
        if not obs:
            return
        pos = np.asarray(self.g.pos, float)
        step = np.asarray(self.g.step)
        pad = self.r + 20.0
        lo = np.array([np.asarray(o.box.min, float) - pad for o in obs])
        hi = np.array([np.asarray(o.box.max, float) + pad for o in obs])
        is_pen = np.array([bool(o.penetrable) for o in obs])
        is_sup = np.array([bool(o.supportable) for o in obs])
        n = pos.shape[0]
        pen = np.zeros((n, N_DIRS), bool)
        sup = np.zeros((n, N_DIRS), bool)
        idx = np.arange(n)
        for d in range(N_DIRS):                       # 방향마다 (n,) — 메모리 상한 고정
            # 사슬 끝 노드 — 모든 노드를 한꺼번에 한 칸씩 민다 (사슬 길이 = 그 선 위 격자점 수)
            cur = idx.copy()
            nxt = step[cur, d]
            live = nxt >= 0
            while live.any():
                cur[live] = nxt[live]
                nxt = step[cur, d]
                live = nxt >= 0
            end = pos[cur]
            for k in range(len(obs)):
                hit = np.all((end >= lo[k]) & (end <= hi[k]), axis=1)
                if is_pen[k]:
                    pen[:, d] |= hit
                if is_sup[k]:
                    sup[:, d] |= hit
        def near(mask: np.ndarray) -> np.ndarray:   # 점 ↔ 박스 최소거리, 반지름 20개 단위
            if not mask.any():
                return np.ones(n, np.float32)
            best = np.full(n, np.inf)
            for k in np.where(mask)[0]:
                gap = np.maximum(np.maximum(lo[k] + pad - pos, pos - hi[k] + pad), 0.0)
                best = np.minimum(best, np.sqrt((gap ** 2).sum(1)))
            return np.minimum(best / (20.0 * self.r), 1.0).astype(np.float32)

        # 상태당 조회를 2번으로 줄인다 — 스칼라 6번 조회는 상태당 +10 µs 였다 (판정이 벽시계, D98)
        node_vec = np.zeros((n, 6), np.float32)
        node_vec[:, 0] = pen.sum(1) / N_DIRS        # 18방향 중 끝에 관통 가능한 것
        node_vec[:, 1] = sup.sum(1) / N_DIRS        # 18방향 중 끝에 지지 가능한 것
        node_vec[:, 4] = near(is_pen)
        node_vec[:, 5] = near(is_sup)
        code = pen.astype(np.uint8) | (sup.astype(np.uint8) << 1)   # (n, 18) 비트 2개
        self.obs_attr = dict(node_vec=node_vec, code=code, pen=pen, sup=sup)

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
        i += 2          # 장애물 속성 예약 (S2) — 0 으로 둔다

        # ── 장애물 기하 ─────────────────────────────────────────────────────
        # **왜 필요한가 (2026-11-09).** 앞 68칸에는 장애물이 **직접 들어가 있지 않았다.**
        # `obs_supportable`·`obs_penetrable` 은 S2 예약으로 항상 0 이고 학습 보고서의
        # 죽은 채널 목록에 올라와 있다. 장애물이 망에 닿는 경로는 `h_norm`(숫자 하나)과
        # 후보 목록의 존재 여부뿐이었다 — "왼쪽에 벽이 있다" 를 망이 볼 수 없었다.
        # §16-6 이 "유일하게 남은 설계 후보" 로 지목한 관측 설계와 같은 자리다.
        v[i:i + N_DIRS] = self.elen_log[s.node]        # 0 = 차단 / 클수록 트였다
        i += N_DIRS
        v[i] = self.n_out_ratio[s.node]                # 열린 출구 비율 (실측 중앙 8/18)
        v[i + 1] = self.clr_r[s.node]                  # 가장 가까운 장애물까지, 반지름 단위
        i += 2

        # ── 관통 창 ─────────────────────────────────────────────────────────
        # 단일 블록이면 `self.win is None` → 6칸 모두 0 (죽은 채널, 의도된 것)
        if self.win is not None:
            cx, wy, wz, hy, hz = self.win
            p = g.pos[s.node]
            dx = (float(p[0]) - cx) / DIAG_MM
            v[i] = 1.0                                 # 이 배관은 관통 배관이다
            v[i + 1] = dx                              # 격벽까지 **부호 있는** x 거리
            v[i + 2] = 1.0 if dx > 0.0 else 0.0        # 이미 건넜는가 (su 가 방금 0 이 된 이유)
            # 창 중심에서 창 반폭 단위로 — |값| ≤ 1 이면 직진으로 창을 통과할 수 있다
            ry = (float(p[1]) - wy) / hy
            rz = (float(p[2]) - wz) / hz
            v[i + 3] = max(-3.0, min(3.0, ry))
            v[i + 4] = max(-3.0, min(3.0, rz))
            v[i + 5] = 1.0 if (abs(ry) <= 1.0 and abs(rz) <= 1.0) else 0.0   # 조준됐다
        i += 6

        # ── 장애물 속성 ─────────────────────────────────────────────────────
        # E1 은 `self.obs_attr is None` → 6칸 모두 0. **죽은 채널이 의도다** — 값이 생기는
        # E2(2단계)와 채널을 더할 수 없게 되는 2→3 freeze 가 같은 단계라, 미루면 넣을 자리가 없다.
        a = self.obs_attr
        if a is not None:
            v[i:i + 6] = a["node_vec"][s.node]         # 관통 비율 · 지지 비율 · (2칸 자리) · 가장 가까운 관통·지지
            c = int(a["code"][s.node, s.dir])          # **지금 가는 방향** 끝에 있는 것
            v[i + 2] = c & 1                           # 관통 가능 → 돌아갈 필요 없다
            v[i + 3] = c >> 1                          # 지지 가능 → 거기 매달면 u 가 0 으로
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
