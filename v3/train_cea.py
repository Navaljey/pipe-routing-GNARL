"""CEA 학습 휴리스틱 — 설계안 §29 (소유자 승인 2026-11-14).

무엇을 배우나
-------------
상태 s 에서 목표까지 남은 **라우터 비용** h*(s) 를, 기존 admissible 휴리스틱 h(s) (D41) 에 대한
**비율 r = h*/h ≥ 1** 의 구간으로 분류한다.

    구간 k = [e_k, e_{k+1}),  e_k = BASE^k   (k = 0 … K−1, 마지막 구간은 위가 열림)
    출력   ĥ = h · e_k̂                       (예측 구간의 **하한**)

* **admissible ⇔ 예측 구간 ≤ 정답 구간** — 논문(Futuhi & Sturtevant, ICLR 2026)의
  "클래스 ≤ 정답 클래스" 와 같은 뜻이다. 구간 0 = [1, BASE) → ĥ = h, 즉 망이 모르면
  **기존 휴리스틱으로 돌아가는** 안전한 바닥이 있다.
* 왜 비율인가 — 논문은 "모든 엣지 비용이 같다"(가정 4)라 정수 비용을 그대로 클래스로 쓴다.
  우리 비용은 연속 kg(수십 ~ 수천 kg)이라 비율의 로그 구간이 크기에 무관한 표현이다.

손실 — CEA (논문 식 5, 구간 색인에 맞춤)
-----------------------------------------
    L = −log( Σ_{k≤t} w_k p_k ) + η · (−log p_t),   w_k = ((k+1)/(t+1))^β

로짓 z 에 대한 기울기는 닫힌 식이다 (a_k = w_k·[k≤t], S = Σ a_k p_k):

    ∂L/∂z = p·(1+η) − a⊙p / S − η·onehot(t)

**numpy 로 구현한다** — 이 환경에서 PyTorch 다운로드 서버가 막혀 있다(403). 망이 작고 기울기가
닫힌 식이라 문제없고, 탐색 중 추론은 원래 numpy 였다.

지표 (MAE 대신, 설계안 §29.3)
------------------------------
① 과대추정 비율 ② 라우트별 max(ĥ − h*) — 정리 3: 품질 손실 ≤ 최적 경로 위 최대 과대추정
③ 강도 = 평균 ĥ/h* (기준: 평균 h/h*) ④ 서포트 비중 밴드별 · 구경별

    python3 v3/train_cea.py --npz labels/cea_feat.npz --out ckpt/cea.npz
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

SUPPORT_BANDS = [(0.0, 0.03), (0.03, 0.09), (0.09, 0.20), (0.20, 0.30), (0.30, 1.01)]


# ── 망 ─────────────────────────────────────────────────────────────────────────

class MLP:
    """입력 → (은닉 ReLU) × depth → K 로짓. 논문과 같은 ReLU (Lipschitz 1)."""

    def __init__(self, dim: int, k: int, hidden: int = 128, depth: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [dim] + [hidden] * depth + [k]
        self.W = [rng.normal(0, math.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    def forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(h)
        return h, acts

    def backward(self, acts, dz):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ dz
            gb[i] = dz.sum(0)
            if i:
                dz = (dz @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    def params(self):
        return self.W + self.b

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params()))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            if self.wd and p.ndim == 2:
                g = g + self.wd * p
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cea_loss_grad(z, t, beta: float, eta: float):
    """CEA 손실과 ∂L/∂z (배치 평균). t: 정답 구간 (n,)."""
    n, k = z.shape
    p = softmax(z)
    kk = np.arange(k)[None, :]
    a = np.where(kk <= t[:, None], ((kk + 1) / (t[:, None] + 1.0)) ** beta, 0.0)
    s = (a * p).sum(1)
    pt = p[np.arange(n), t]
    loss = float((-np.log(s + 1e-12) - eta * np.log(pt + 1e-12)).mean())
    g = p * (1.0 + eta) - a * p / (s[:, None] + 1e-12)
    g[np.arange(n), t] -= eta
    return loss, (g / n).astype(np.float32)


# ── 구간 ────────────────────────────────────────────────────────────────────────

def ratio_class(r: np.ndarray, base: float, k: int) -> np.ndarray:
    return np.clip(np.floor(np.log(np.maximum(r, 1.0)) / math.log(base)), 0, k - 1).astype(np.int64)


def decode(p: np.ndarray, mode: str) -> np.ndarray:
    """확률 → 구간. argmax = 논문 방식. q<x> = 누적확률이 x 이상인 **가장 낮은** 구간 (더 보수적)."""
    if mode == "argmax":
        return p.argmax(1)
    q = float(mode[1:])
    return (np.cumsum(p, 1) >= q).argmax(1)


# ── 평가 ────────────────────────────────────────────────────────────────────────

def evaluate(net, Xn, h, hstar, t, route, sup_share, size, base, mode="argmax"):
    z, _ = net.forward(Xn)
    kh = decode(softmax(z), mode)
    hh = h * base ** kh
    over = kh > t
    gap = hh - hstar
    per_route_max = {}
    for r in np.unique(route):
        m = route == r
        per_route_max[int(r)] = float(gap[m].max())
    rmax = np.asarray(list(per_route_max.values()))
    out = {
        "decode": mode, "n": int(len(t)),
        "over_rate": float(over.mean()),
        "routes_with_over": int((rmax > 1e-6).sum()), "n_routes": int(len(rmax)),
        "route_max_over_kg_max": float(max(rmax.max(), 0.0)),
        "route_max_over_rel_max": float(max(((hh - hstar) / hstar)[over].max(), 0.0)) if over.any() else 0.0,
        "strength_cea": float((hh / hstar).mean()),
        "strength_h": float((h / hstar).mean()),
        "exact_class": float((kh == t).mean()),
    }
    bands = {}
    for lo, hi in SUPPORT_BANDS:
        m = (sup_share >= lo) & (sup_share < hi)
        if m.sum() >= 20:
            bands[f"{lo:.2f}-{hi:.2f}"] = {"n": int(m.sum()), "over_rate": float(over[m].mean()),
                                           "strength_cea": float((hh[m] / hstar[m]).mean()),
                                           "strength_h": float((h[m] / hstar[m]).mean())}
    out["by_support_band"] = bands
    sizes = {}
    for s in np.unique(size):
        m = size == s
        if m.sum() >= 20:
            sizes[str(s)] = {"n": int(m.sum()), "over_rate": float(over[m].mean()),
                             "strength_cea": float((hh[m] / hstar[m]).mean()),
                             "strength_h": float((h[m] / hstar[m]).mean())}
    out["by_size"] = sizes
    return out


def split_by_route(route, frac, seed):
    ids = np.unique(route)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    val = set(ids[:max(1, int(round(len(ids) * frac)))].tolist())
    m = np.array([r in val for r in route])
    return ~m, m


# ── 추론 (탐색에서 쓴다) ─────────────────────────────────────────────────────────

class CEAHeuristic:
    """체크포인트 → 상태 특징 1개로 ĥ 를 낸다. 탐색은 max(h, ĥ) 를 쓴다."""

    def __init__(self, path: str, mode: str = "argmax"):
        d = np.load(path, allow_pickle=True)
        nl = int(d["n_layers"])
        self.W = [d[f"W{i}"] for i in range(nl)]
        self.b = [d[f"b{i}"] for i in range(nl)]
        self.mu, self.sd = d["mu"], d["sd"]
        self.base, self.k = float(d["base"]), int(d["k"])
        self.dim = int(d["dim"])
        self.mode = mode
        self.support_standard = str(d["support_standard"])

    def ratio(self, x: np.ndarray) -> float:
        h = (x[: self.dim] - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        if self.mode == "argmax":
            k = int(h.argmax())
        else:
            p = np.exp(h - h.max())
            p /= p.sum()
            k = int((np.cumsum(p) >= float(self.mode[1:])).argmax())
        return self.base ** k


# ── 학습 ────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CEA 학습 휴리스틱 (설계안 §29)")
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=float, default=1.05, help="구간 비율 폭 (5%%)")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3, help="과대추정 미개선 n회면 β·η 절반 (부록 J)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    parts = [np.load(p, allow_pickle=True) for p in a.npz]
    cat = lambda key: np.concatenate([d[key] for d in parts])          # noqa: E731
    X = cat("X").astype(np.float32)
    h = cat("h").astype(np.float64)
    hstar = (cat("y_pe") + cat("y_sup") + cat("y_pen")).astype(np.float64)
    sup_share = cat("y_sup") / np.maximum(hstar, 1e-9)
    size = cat("size")
    # 라우트 번호는 파일마다 0 부터라 섞이지 않게 (파일, 라우트) 로 다시 매긴다
    route = np.concatenate([d["route"].astype(np.int64) + 1_000_000 * i for i, d in enumerate(parts)])
    names = list(parts[0]["names"])
    std = str(parts[0]["support_standard"][0])

    keep = (h > 1e-9) & (hstar > 1e-9)
    r = hstar / np.where(h > 1e-9, h, 1.0)
    n_inadm = int((keep & (r < 1 - 1e-6)).sum())
    print(f"상태 {len(h)} → 학습 대상 {keep.sum()} (목표·h=0 제외) · "
          f"h > h* 인 상태 {n_inadm} (≠ 0 이면 h 가 admissible 이 아니다 — 확인 필요)")
    X, h, hstar, sup_share, size, route, r = (v[keep] for v in (X, h, hstar, sup_share, size, route, r))
    t = ratio_class(r, a.base, a.k)
    cnt = np.bincount(t, minlength=a.k)
    print(f"비율 h*/h — 중앙 {np.median(r):.3f} · p90 {np.quantile(r, .9):.3f} · 최대 {r.max():.2f} · "
          f"마지막 구간(열림) {cnt[-1]} 상태")

    tr, va = split_by_route(route, a.val_frac, a.seed)
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-6] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    net = MLP(X.shape[1], a.k, a.hidden, a.depth, a.seed)
    opt = Adam(net.params(), a.lr, wd=a.wd)
    print(f"라우트 학습 {len(np.unique(route[tr]))} · 검증 {len(np.unique(route[va]))} · "
          f"상태 {tr.sum()} / {va.sum()} · 파라미터 {net.n_params():,}")

    rng = np.random.default_rng(a.seed)
    beta, eta = a.beta, a.eta
    best_over, stall = math.inf, 0
    hist = []
    t0 = time.perf_counter()
    idx_tr = np.where(tr)[0]
    for ep in range(1, a.epochs + 1):
        rng.shuffle(idx_tr)
        tot = 0.0
        for s in range(0, len(idx_tr), a.batch):
            bi = idx_tr[s:s + a.batch]
            z, acts = net.forward(Xn[bi])
            loss, dz = cea_loss_grad(z, t[bi], beta, eta)
            gW, gb = net.backward(acts, dz)
            opt.step(gW + gb)
            tot += loss * len(bi)
        if ep % a.eval_every == 0 or ep == 1:
            m = evaluate(net, Xn[va], h[va], hstar[va], t[va], route[va], sup_share[va], size[va], a.base)
            m.update(epoch=ep, train_loss=tot / len(idx_tr), beta=beta, eta=eta)
            hist.append({k: v for k, v in m.items() if not k.startswith("by_")})
            print(f"  ep {ep:3d} loss {m['train_loss']:.4f} · 과대추정 {m['over_rate']:.4f} "
                  f"(라우트 {m['routes_with_over']}/{m['n_routes']}) · 강도 {m['strength_cea']:.3f} "
                  f"(h {m['strength_h']:.3f}) · 정확구간 {m['exact_class']:.3f} · β {beta:g} η {eta:g}",
                  flush=True)
            if m["over_rate"] < best_over - 1e-9:
                best_over, stall = m["over_rate"], 0
            else:
                stall += 1
                if stall >= a.patience:
                    beta, eta, stall = beta / 2, eta / 2, 0

    fin = {mode: evaluate(net, Xn[va], h[va], hstar[va], t[va], route[va], sup_share[va], size[va],
                          a.base, mode) for mode in ("argmax", "q0.1", "q0.05")}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrs = {f"W{i}": W for i, W in enumerate(net.W)} | {f"b{i}": b for i, b in enumerate(net.b)}
    np.savez(out, **arrs, n_layers=len(net.W), mu=mu, sd=sd, base=a.base, k=a.k, dim=X.shape[1],
             names=np.asarray(names), support_standard=np.asarray(std), beta=beta, eta=eta)
    out.with_suffix(".report.json").write_text(json.dumps(
        {"final": fin, "hist": hist, "class_hist": cnt.tolist(), "n_inadmissible_h": n_inadm,
         "args": vars(a), "n_params": net.n_params(), "sec": round(time.perf_counter() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, m in fin.items():
        print(f"\n[{mode}] 과대추정 {m['over_rate']:.4f} · 과대추정 있는 라우트 {m['routes_with_over']}/"
              f"{m['n_routes']} · 최대 과대추정 {m['route_max_over_kg_max']:.1f} kg "
              f"(상대 {m['route_max_over_rel_max']:.3f}) · 강도 {m['strength_cea']:.3f} (h {m['strength_h']:.3f})")
    print(f"\n{out} · {round(time.perf_counter() - t0)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
