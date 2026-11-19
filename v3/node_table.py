"""C안 — TransPath 식 "맵당 1회 추론" 노드 표 (2026-11-19 소유자 지시 "C 방법을 측정해줘").

근거: Kirilenko 외, *TransPath: Learning Heuristics for Grid-Based Pathfinding via Transformers* (AAAI 2023).
인스턴스 독립 휴리스틱에 대한 **보정 계수(correction factor)** 를 맵 전체에 대해 **한 번에** 예측해 두고,
탐색은 그 표를 조회만 한다.

우리 쪽 번역
------------
* 인스턴스 독립 휴리스틱 = `rg` (완화 그래프 정확 거리, §30.2). 보정 계수 = `h*/rg` (≥ 1).
* §30.10 `cearg` 와 **같은 표현·손실** (비율 구간 1.05^k, CEA, q0.05 디코딩) — 다른 것은 두 가지뿐:
    ① 입력이 **상태가 아니라 노드** 특징이다 (방향·run·su·hz 를 모른다)
    ② 추론을 **탐색 전에 노드 전체(유효 부분집합)에 대해 1회** 배치로 한다 → 탐색 중 추론 비용 0
* 표 `htab[n] = rg[n] · 1.05^k̂(n)` 을 skeleton 컴파일 A\\* 패치(`v3.astar_rg.route_rg`)에 그대로 넣는다.
  탐색 휴리스틱 = max(skeleton h(상태), htab[노드]).
* 추론 비용(특징 + MLP)은 `rg` 사전 계산과 함께 **벽시계에 더한다** (D98).

TransPath 와 다른 점 (솔직히): 원 논문은 CNN+트랜스포머로 격자 전체를 보지만 여기는 **노드별 MLP**다.
장거리 맥락은 `rg` 값과 그 하강 방향이 대신 실어 나른다. "맵당 1회 추론으로 탐색 중 추론 비용을
없앤다" 는 C안의 핵심만 재는 최소판이다.

추론 부분집합 (측정 전 선언): `rg[n] ≤ 3 · rg[start]` 인 노드만 예측하고 나머지는 `htab = rg`.
A\\* 가 확장하는 노드는 g + h ≤ C\\* 이고 g ≥ 0 이므로 rg ≤ C\\* 이다. C\\*/rg(start) 는 p90 2.38 (§30.9) —
3 배 밖 노드는 거의 확장되지 않는다. 밖에서도 `htab = rg` 라 admissible 하한은 그대로다.
"""

from __future__ import annotations

import math
import time

import numpy as np

from v3.features import DIAG_MM, KG_PER_M_MAX, N_DIRS, R_MAX, FeatureMaker
from pipe_routing.pipe_spec import pipe_spec
from pipe_routing.space import DIRS

SUBSET_FACTOR = 3.0

NAMES = (["goal_dx", "goal_dy", "goal_dz", "goal_dist", "pos_x", "pos_y", "pos_z",
          "rg_norm", "log_rg_over_euclid"]
         + [f"elen_{i}" for i in range(N_DIRS)] + ["n_out_ratio", "clearance_r"]
         + ["sup_out_frac", "sup_out_mean"]
         + [f"rgdesc_{i}" for i in range(N_DIRS)] + ["desc_dot_end", "desc_dot_goal"]
         + [f"enddir_{i}" for i in range(N_DIRS)]
         + ["kg_per_m", "elbow_in_m", "radius", "turn_need_90", "bend_u_90", "is_gravity"])
DIM = len(NAMES)

_DIRS = np.asarray(DIRS, float)
_DIRS_U = _DIRS / np.linalg.norm(_DIRS, axis=1, keepdims=True)


def node_features(g, pipe, rg: np.ndarray, idx: np.ndarray, fm: FeatureMaker | None = None) -> np.ndarray:
    """노드 색인 `idx` 의 특징 (len(idx), DIM). 전부 벡터 연산 — 상태를 모른다."""
    fm = fm or FeatureMaker(g, pipe)
    spec = pipe_spec(pipe)
    kgpm = spec.kg_per_m
    n = len(idx)
    X = np.zeros((n, DIM), np.float32)
    p = np.asarray(g.pos, float)[idx]
    d = fm.goal[None, :] - p
    dist = np.linalg.norm(d, axis=1)
    X[:, 0:3] = d / DIAG_MM
    X[:, 3] = dist / DIAG_MM
    X[:, 4:7] = p / DIAG_MM
    r = rg[idx]
    X[:, 7] = r / fm.j_scale
    eu = dist / 1000.0 * kgpm
    X[:, 8] = np.log(np.maximum(r, 1e-6) / np.maximum(eu, 1e-6)).clip(-2, 4)
    i = 9
    X[:, i:i + N_DIRS] = fm.elen_log[idx]
    i += N_DIRS
    X[:, i] = fm.n_out_ratio[idx]
    X[:, i + 1] = fm.clr_r[idx]
    i += 2
    step = g.step[idx]
    openm = (step >= 0) & (g.emask[idx] != 0)
    sc = g.scost[idx]
    n_open = np.maximum(openm.sum(1), 1)
    X[:, i] = ((sc > 0) & openm).sum(1) / n_open
    X[:, i + 1] = np.where(openm, sc, 0.0).sum(1) / n_open / max(spec.elbow90, 1e-6)
    i += 2
    # rg 하강 방향 — 완화 최단 경로가 이 노드에서 처음 가는 방향 (장거리 맥락의 대리)
    w = g.elen[idx] / 1000.0 * kgpm + sc
    nxt = np.where(step >= 0, step, 0)
    val = np.where(openm, w + rg[nxt], np.inf)
    best = val.argmin(1)
    has = np.isfinite(val.min(1))
    X[np.arange(n)[has], i + best[has]] = 1.0
    i += N_DIRS
    bu = _DIRS_U[best]
    if fm.end_dir_idx >= 0:
        X[:, i] = np.where(has, (bu * _DIRS_U[fm.end_dir_idx]).sum(1), 0.0)
    du = d / np.maximum(dist, 1e-6)[:, None]
    X[:, i + 1] = np.where(has, (bu * du).sum(1), 0.0)
    i += 2
    if fm.end_dir_idx >= 0:
        X[:, i + fm.end_dir_idx] = 1.0
    i += N_DIRS
    X[:, i] = kgpm / KG_PER_M_MAX
    X[:, i + 1] = spec.elbow90 / kgpm
    X[:, i + 2] = (spec.od / 2) / R_MAX
    X[:, i + 3] = fm.turn90 / DIAG_MM
    X[:, i + 4] = fm.bend_u90
    X[:, i + 5] = 1.0 if fm.is_grav else 0.0
    assert i + 6 == DIM
    return X


class NodeCF:
    """체크포인트(train_cea 형식) → 노드 표. 배치 1회 추론."""

    def __init__(self, path: str, mode: str = "q0.05"):
        d = np.load(path, allow_pickle=True)
        nl = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(nl)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(nl)]
        self.mu, self.sd = d["mu"].astype(np.float32), d["sd"].astype(np.float32)
        self.base, self.k = float(d["base"]), int(d["k"])
        assert int(d["dim"]) == DIM, "노드 특징 차원이 체크포인트와 다르다"
        self.mode = mode

    def ratios(self, X: np.ndarray) -> np.ndarray:
        h = (X - self.mu) / self.sd
        for j, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if j < len(self.W) - 1:
                np.maximum(h, 0.0, out=h)
        if self.mode == "argmax":
            k = h.argmax(1)
        else:
            h -= h.max(1, keepdims=True)
            p = np.exp(h)
            p /= p.sum(1, keepdims=True)
            k = (np.cumsum(p, 1) >= float(self.mode[1:])).argmax(1)
        return self.base ** k

    def table(self, g, pipe, rg: np.ndarray, start_node: int) -> tuple[np.ndarray, dict]:
        """→ (htab, 통계). 시간은 호출자가 잰다 (특징 + 추론 전부 포함)."""
        lim = SUBSET_FACTOR * float(rg[start_node])
        idx = np.nonzero(np.isfinite(rg) & (rg > 0) & (rg <= lim))[0]
        t = time.perf_counter()
        X = node_features(g, pipe, rg, idx)
        t1 = time.perf_counter()
        ratio = self.ratios(X)
        t2 = time.perf_counter()
        htab = rg.copy()
        htab[idx] = rg[idx] * ratio
        return htab, {"n_pred": int(len(idx)), "n_nodes": int(len(rg)),
                      "feat_sec": round(t1 - t, 4), "mlp_sec": round(t2 - t1, 4),
                      "ratio_mean": float(ratio.mean()) if len(idx) else 1.0}
