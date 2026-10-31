"""순위 학습 — **"후보 집합에서 하나 고르기"** 를 직접 배운다 (소유자 지시 2026-11-01, 2c).

회귀와 무엇이 다른가
--------------------
`train_h.py` 는 상태마다 **남은 kg** 을 맞혔다. 그 값이 FOCAL 안 순서를 정하기를
기대했는데, ε=0.5 에서 R = 0.637 (기준선보다 느림) 이었다 — 오라클 104.8 대비 **165배**.
원인은 라벨이 **최적 경로 위 상태뿐**이라 띠 안 경로 밖 후보에서 외삽한다는 것 (§16.1②).

여기서는 라벨이 **탐색이 실제로 마주친 결정**이다:

    한 결정 = (후보 n 개의 특징,  오라클이 고른 슬롯)

손실 = 후보 집합 위 **softmax 교차 엔트로피** (listwise). 점수가 낮을수록 먼저 고르므로
`logits = -score` 로 둔다 — 추론 때 `priority = score` 그대로 쓰면 된다.

왜 교차 엔트로피인가: 우리가 쓰는 연산이 정확히 **argmin** 이다. "정답에 가장 낮은 점수를"
가 목적이지 "정답의 값을 맞히기" 가 아니다. §17.3 에서 비용 회귀 지표가 질문에 안 맞았던
것의 교정이다.

패딩
----
후보 수는 매번 다르다 (띠가 좁은 순간은 N_CAND 보다 적다). 라벨은 `N_CAND` 로 0 패딩하고
`n_valid` 를 적어 두므로, **손실에서 유효 슬롯 밖을 −inf 로 가린다.** 가리지 않으면
0 벡터가 후보로 들어가 가짜 경쟁자가 된다.

주 지표
-------
* **top-1 일치율** — 오라클이 고른 것을 1순위로 꼽는 비율. 이것이 탐색에서 쓰이는 양이다.
* **정답 순위 중앙** — 틀렸을 때 얼마나 틀렸나 (1 이면 맞음).
* 둘 다 **n_focal 밴드별**로 쪼갠다 — 띠가 넓을수록 어려우므로, 넓은 밴드에서 못 맞히면
  정확히 우리가 겨냥한 구간에서 실패한 것이다.

    python3 v3/train_rank.py --npz labels/order_eps_0.5.npz --out ckpt/rank_eps05.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

FOCAL_BANDS = [(0, 100), (100, 1000), (1000, 10000), (10000, 10**9)]


class RankNet(nn.Module):
    """상태 → 점수 (작을수록 먼저). `train_h.HNet` 과 같은 몸통 폭·깊이를 쓸 수 있게 둔다."""

    def __init__(self, dim: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.flops = sum(m.in_features * m.out_features * 2 for m in self.modules()
                         if isinstance(m, nn.Linear))

    def forward(self, x):            # x: (..., dim) → (...)
        return self.head(self.trunk(x)).squeeze(-1)


def split_by_route(route: np.ndarray, frac: float, seed: int = 0):
    """**라우트 단위** 분할 — 같은 라우트의 결정들은 서로 극도로 닮았다 (같은 경로 주변)."""
    ids = np.unique(route)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    val = set(ids[:max(1, int(round(len(ids) * frac)))].tolist())
    m = np.array([r in val for r in route])
    return ~m, m


def evaluate(net, X, nv, gold, n_focal, bs=512):
    """top-1 일치율 · 정답 순위 중앙 · n_focal 밴드별."""
    net.eval()
    ranks = []
    with torch.no_grad():
        for k in range(0, len(gold), bs):
            xb = torch.from_numpy(X[k:k + bs])
            sb = net(xb).numpy()
            for i in range(len(sb)):
                v = int(nv[k + i])
                s = sb[i, :v]
                g = int(gold[k + i])
                ranks.append(int((s < s[g]).sum()) + 1)   # 정답보다 낮은 점수 개수 + 1
    ranks = np.asarray(ranks)
    out = {"top1": float((ranks == 1).mean()), "rank_median": float(np.median(ranks)),
           "n": int(len(ranks))}
    for lo, hi in FOCAL_BANDS:
        m = (n_focal >= lo) & (n_focal < hi)
        if m.sum() >= 20:
            out[f"top1_focal_{lo}_{hi}"] = float((ranks[m] == 1).mean())
            out[f"n_focal_{lo}_{hi}"] = int(m.sum())
    return out, ranks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FOCAL 안 순위 학습")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    d = np.load(a.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)                 # (N, N_CAND, DIM)
    gold = d["gold"].astype(np.int64)
    nv = d["n_valid"].astype(np.int64)
    route = d["route"]
    n_focal = d["n_focal"]
    names = list(d["names"])
    eps = float(d["eps"][0])
    print(f"ε={eps} · 결정 {len(gold)} · 후보칸 {X.shape[1]} · 특징 {X.shape[2]} · "
          f"라우트 {len(np.unique(route))}")
    print(f"유효 후보 수 중앙 {int(np.median(nv))} · |FOCAL| 중앙 {int(np.median(n_focal))}")

    tr, va = split_by_route(route, a.val_frac, a.seed)
    # 입력 정규화 — 유효 슬롯만으로 통계를 낸다 (0 패딩이 평균을 끌어내리지 않게)
    mask_tr = np.arange(X.shape[1])[None, :] < nv[tr][:, None]
    flat = X[tr][mask_tr]
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-6] = 1.0
    dead = [names[i] for i in range(X.shape[2]) if flat[:, i].std() == 0]
    if dead:
        print(f"⚠️  학습셋 유효 후보에서 상수인 채널 {len(dead)}개: {dead}")
    Xn = ((X - mu) / sd).astype(np.float32)
    Xn[np.arange(X.shape[1])[None, :] >= nv[:, None]] = 0.0   # 패딩 자리는 0 유지

    net = RankNet(X.shape[2], a.hidden, a.depth)
    print(f"망 hidden {a.hidden} × depth {a.depth} · 파라미터 "
          f"{sum(p.numel() for p in net.parameters())} · 상태당 약 {net.flops:,} FLOPs")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    Xtr, Xva = Xn[tr], Xn[va]
    gtr = torch.from_numpy(gold[tr])
    ntr = torch.from_numpy(nv[tr])
    slots = torch.arange(X.shape[1])
    n = len(gtr)
    hist: list[dict] = []
    t0 = time.perf_counter()

    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, a.batch):
            idx = perm[k:k + a.batch]
            xb = torch.from_numpy(Xtr[idx.numpy()])
            score = net(xb)                                   # (B, N_CAND) 작을수록 먼저
            # 유효 슬롯 밖은 경쟁에서 제외 — 안 가리면 0 벡터가 가짜 후보가 된다
            invalid = slots[None, :] >= ntr[idx][:, None]
            logits = (-score).masked_fill(invalid, float("-inf"))
            loss = nn.functional.cross_entropy(logits, gtr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        if ep % 5 == 0 or ep == 1:
            m, _ = evaluate(net, Xva, nv[va], gold[va], n_focal[va])
            m.update(epoch=ep, train_loss=tot / n)
            hist.append(m)
            bands = " · ".join(f"{k.replace('top1_focal_', '|F|')}={v:.3f}"
                               for k, v in m.items() if k.startswith("top1_focal_"))
            print(f"  ep {ep:3d} loss {m['train_loss']:.4f} · top-1 {m['top1']:.3f} · "
                  f"정답순위 중앙 {m['rank_median']:.0f} · {bands}", flush=True)

    # 기준점 — 학습 없이 `h` 순(= f 순, 지금 포컬의 기본)으로 고르면 얼마인가
    hi = names.index("h_norm")
    base_rank = []
    for i in np.where(va)[0]:
        v = int(nv[i])
        s = X[i, :v, hi]
        base_rank.append(int((s < s[gold[i]]).sum()) + 1)
    base_rank = np.asarray(base_rank)
    base_top1 = float((base_rank == 1).mean())

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "names": names,
                "dim": X.shape[2], "hidden": a.hidden, "depth": a.depth, "eps": eps,
                "flops_per_state": net.flops, "kind": "rank", "hist": hist,
                "support_standard": str(d["support_standard"][0])}, out)
    last = hist[-1]
    out.with_suffix(".report.json").write_text(json.dumps(
        {"last": last, "baseline_h_top1": base_top1, "dead_channels": dead,
         "eps": eps, "n_decisions": int(len(gold)), "hist": hist},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out} · {round(time.perf_counter() - t0)}s")
    print(f"최종 top-1 {last['top1']:.3f} (h 순 기준점 {base_top1:.3f}) · "
          f"정답 순위 중앙 {last['rank_median']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
