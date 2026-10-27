"""1단계 — `ĥ` 2머리 지도학습 (D-p 확정 · 서포트 학습 감시, 소유자 지시 2026-10-28 §5).

구조
----
    ĥ_pipe = h + softplus(머리₁)       목표 rem_pe    (직관 + 엘보)
    ĥ_sup  =     softplus(머리₂)       목표 rem_sup   (추정 서포트 D57)
    ĥ      = ĥ_pipe + ĥ_sup            FOCAL 순서는 **합**으로 (kg 단위 유지)

`h`(D41) 위의 **비음수 잔차**를 배운다. `h` 가 admissible 이라 `rem_pe ≥ h` 가 항상
참이므로 softplus 로 비음수를 강제하면 ĥ ≥ h 가 보장된다 — 망이 이미 아는 부분을 다시
배우지 않고 **`h` 의 맹점(서포트·구배)에만** 용량을 쓴다.

왜 2머리인가 (소유자 우려의 산수)
---------------------------------
서포트는 라우터 비용의 8~9% 다 (§10.6). 단일 머리에 MSE 를 걸면 **서포트가 기울기의
8~9% 뿐**이라 배관 항이 먼저, 서포트가 나중에 배운다 — 소유자가 지적한 바로 그것이다.
→ 머리를 쪼개고 **성분별 척도로 정규화해** 절반씩 가게 한다.

    loss = (잔차_pipe 오차 / s_p)² + w_s · (서포트 오차 / s_s)²
    s_p, s_s = 학습셋의 성분별 표준편차 (고정, 체크포인트에 기록)

`w_s = 1` 이 기본이다 (정규화 후 동등). `w_s > 1` 은 **측정 후에** 정한다.

보고 (D-v · 설계안 §14.3~14.4)
------------------------------
* `MAE_pipe` · `MAE_sup` · 정규화 `MAE_sup/s_s` 를 **따로** 기록한다.
* **서포트 비중 밴드별 · 구경별**로 쪼갠다 — 서포트 머리가 작동하면 비중 높은 밴드에서
  더 개선돼야 한다. **반증 가능한 예측**이다.
* **주 지표는 MAE 가 아니라 FOCAL 안 순위 일치율**이다 (§9 결론: "띠 안 순서가 전부").
  여기서는 라벨만으로 재는 대리 지표를 쓴다 — 같은 라우트의 **경로 위 상태 쌍**에서
  ĥ 가 rem_total 의 순서를 맞추는 비율 (쌍별 순위 일치, Kendall 류). 진짜 FOCAL 순위는
  탐색을 돌려야 하므로 `v3/measure_rank.py` 가 따로 잰다.
* **정체 기준:** 연속 3회 평가에서 `MAE_sup/s_s` 가 상대 2% 미개선인데 `MAE_pipe` 는
  개선 → **"서포트 머리 정체"** 로 보고한다.

    python3 v3/train_h.py --npz labels/stage1_feat.npz --out ckpt/h_stage1.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

SUPPORT_BANDS = [(0.0, 0.03), (0.03, 0.09), (0.09, 0.20), (0.20, 0.30), (0.30, 1.01)]


class HNet(nn.Module):
    """2머리. 공유 몸통 + 머리 2개 — 서포트와 배관이 같은 기하를 보지만 출력은 분리한다."""

    def __init__(self, dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)      # depth=0 → 선형 모형 (배포 비용 하한)
        self.head_pipe = nn.Linear(d, 1)
        self.head_sup = nn.Linear(d, 1)
        #: 상태 1개 추론 FLOPs — 벽시계 판정(D98)에서 이게 예산을 먹는다. §추론 예산 참조
        self.flops = sum(m.in_features * m.out_features * 2 for m in self.modules()
                         if isinstance(m, nn.Linear))

    def forward(self, x, h):
        z = self.trunk(x)
        resid = nn.functional.softplus(self.head_pipe(z).squeeze(-1))    # ≥ 0 → ĥ_pipe ≥ h
        sup = nn.functional.softplus(self.head_sup(z).squeeze(-1))       # ≥ 0
        return h + resid, sup


def split_by_route(route: np.ndarray, frac: float, seed: int = 0):
    """**라우트 단위**로 쪼갠다. 상태 단위로 쪼개면 같은 경로의 이웃 상태가 양쪽에
    들어가 누수가 된다 (상태 간 상관이 극도로 높다)."""
    ids = np.unique(route)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * frac)))
    val = set(ids[:n_val].tolist())
    m = np.array([r in val for r in route])
    return ~m, m


def pair_rank_agreement(pred: np.ndarray, true: np.ndarray, route: np.ndarray,
                        rng: np.random.Generator, n_pairs: int = 20000) -> float:
    """같은 라우트 안에서 무작위 상태 쌍을 뽑아 순서 일치율. 대리 지표 (주 지표의 예고편)."""
    order = np.argsort(route, kind="stable")
    r = route[order]
    bounds = np.searchsorted(r, np.unique(r), side="left").tolist() + [len(r)]
    ok = tot = 0
    for _ in range(n_pairs):
        k = rng.integers(0, len(bounds) - 1)
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < 2:
            continue
        i, j = rng.integers(lo, hi), rng.integers(lo, hi)
        if i == j or abs(true[order[i]] - true[order[j]]) < 1e-6:
            continue
        tot += 1
        ok += (pred[order[i]] < pred[order[j]]) == (true[order[i]] < true[order[j]])
    return ok / tot if tot else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ĥ 2머리 지도학습")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="ckpt/h_stage1.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--w-sup", type=float, default=1.0, help="정규화 후 서포트 손실 가중 (기본 1 = 동등)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-head", action="store_true",
                    help="대조군 — 합계 1머리. 서포트가 기울기의 8~9% 뿐이라는 것을 보이는 용도")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    d = np.load(a.npz, allow_pickle=True)
    X = d["X"].astype(np.float32)
    h = d["h"].astype(np.float32)
    y_pe = d["y_pe"].astype(np.float32)
    y_sup = d["y_sup"].astype(np.float32)
    route = d["route"]
    size = d["size"]
    names = list(d["names"])
    share = y_sup / np.maximum(y_pe + y_sup, 1e-9)

    tr, va = split_by_route(route, a.val_frac, a.seed)
    print(f"특징 {X.shape[1]} · 상태 학습 {tr.sum()} / 검증 {va.sum()} · "
          f"라우트 {len(np.unique(route))} ({len(np.unique(route[va]))} 검증)")

    # 입력 정규화 — 학습셋 통계로 고정하고 체크포인트에 넣는다 (추론 때 같은 값을 써야 한다)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-6] = 1.0
    dead = [names[i] for i in range(X.shape[1]) if d["X"][tr][:, i].std() == 0]
    if dead:
        print(f"⚠️  학습셋에서 상수인 채널 {len(dead)}개: {dead}\n"
              "    → 이 채널은 학습되지 않는다. 탐색 중 값이 변하면 외삽이다 (정직하게 기록)")

    s_p = float((y_pe[tr] - h[tr]).std()) or 1.0     # 잔차 척도
    s_s = float(y_sup[tr].std()) or 1.0              # 서포트 척도
    print(f"척도 s_p(잔차) {s_p:.2f} kg · s_s(서포트) {s_s:.2f} kg · "
          f"서포트 비중 중앙 {np.median(share[tr]):.3f}")

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x))

    Xn = (X - mu) / sd
    Xtr, Xva = T(Xn[tr]), T(Xn[va])
    htr, hva = T(h[tr]), T(h[va])
    ptr, pva = T(y_pe[tr]), T(y_pe[va])
    str_, sva = T(y_sup[tr]), T(y_sup[va])

    net = HNet(X.shape[1], a.hidden, a.depth)
    print(f"망 hidden {a.hidden} × depth {a.depth} · 파라미터 "
          f"{sum(p.numel() for p in net.parameters())} · 상태당 약 {net.flops:,} FLOPs")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    n = int(tr.sum())
    rng = np.random.default_rng(a.seed)
    hist: list[dict] = []
    stall_run = 0
    t0 = time.perf_counter()

    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, a.batch):
            idx = perm[k:k + a.batch]
            hp, hs = net(Xtr[idx], htr[idx])
            if a.single_head:
                # 대조군: 합계에만 MSE. 서포트는 합의 8~9% 라 기울기도 그만큼이다
                loss = (((hp + hs) - (ptr[idx] + str_[idx])) / (s_p + s_s)).pow(2).mean()
            else:
                loss = (((hp - ptr[idx]) / s_p).pow(2).mean()
                        + a.w_sup * ((hs - str_[idx]) / s_s).pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()

        net.eval()
        with torch.no_grad():
            hp, hs = net(Xva, hva)
            hp, hs = hp.numpy(), hs.numpy()
        mae_p = float(np.abs(hp - y_pe[va]).mean())
        mae_s = float(np.abs(hs - y_sup[va]).mean())
        tot_pred, tot_true = hp + hs, y_pe[va] + y_sup[va]
        mae_t = float(np.abs(tot_pred - tot_true).mean())
        # 기준점 — h 를 그대로 쓰면 (현재 포컬 탐색의 f) 얼마인가
        base_t = float(np.abs(h[va] - tot_true).mean())
        rank = pair_rank_agreement(tot_pred, tot_true, route[va], rng)
        rank_h = pair_rank_agreement(h[va], tot_true, route[va], rng)
        row = {"epoch": ep, "train_loss": tot / n, "MAE_pipe": mae_p, "MAE_sup": mae_s,
               "MAE_sup_norm": mae_s / s_s, "MAE_total": mae_t, "MAE_total_h_only": base_t,
               "rank_agree": rank, "rank_agree_h_only": rank_h}
        # 서포트 비중 밴드별 (반증 가능한 예측: 비중 높은 밴드에서 더 개선돼야 한다)
        for lo, hi in SUPPORT_BANDS:
            m = (share[va] >= lo) & (share[va] < hi)
            if m.sum() >= 20:
                row[f"MAE_sup_band_{lo:.2f}_{hi:.2f}"] = float(np.abs(hs[m] - y_sup[va][m]).mean())
        hist.append(row)

        if ep >= 4:                                   # 정체 기준 (설계안 §14.4)
            prev = hist[-2]
            imp_s = (prev["MAE_sup_norm"] - row["MAE_sup_norm"]) / max(prev["MAE_sup_norm"], 1e-9)
            imp_p = (prev["MAE_pipe"] - row["MAE_pipe"]) / max(prev["MAE_pipe"], 1e-9)
            stall_run = stall_run + 1 if (imp_s < 0.02 and imp_p > 0) else 0
            if stall_run == 3:
                print(f"🔻 서포트 머리 정체 (ep {ep}): MAE_sup/s_s 3회 연속 상대 2% 미개선, "
                      f"MAE_pipe 는 개선 중")
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d} loss {row['train_loss']:.4f} · MAE_pipe {mae_p:6.2f} · "
                  f"MAE_sup {mae_s:6.2f} ({row['MAE_sup_norm']:.3f}) · MAE_tot {mae_t:6.2f} "
                  f"(h 단독 {base_t:6.2f}) · 순위 {rank:.3f} (h 단독 {rank_h:.3f})", flush=True)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "s_p": s_p, "s_s": s_s,
                "names": names, "dim": X.shape[1], "hidden": a.hidden, "depth": a.depth,
                "single_head": a.single_head, "w_sup": a.w_sup, "flops_per_state": net.flops,
                "support_standard": str(d["support_standard"][0]),
                "hist": hist}, out)
    last = hist[-1]
    # 구경별 (D-v)
    with torch.no_grad():
        hp, hs = net(Xva, hva)
    per_size = {}
    for sz in sorted(set(size[va].tolist())):
        m = size[va] == sz
        if m.sum() >= 20:
            per_size[sz] = {"n": int(m.sum()),
                            "MAE_pipe": round(float(np.abs(hp.numpy()[m] - y_pe[va][m]).mean()), 3),
                            "MAE_sup": round(float(np.abs(hs.numpy()[m] - y_sup[va][m]).mean()), 3),
                            "share_median": round(float(np.median(share[va][m])), 4)}
    (out.with_suffix(".report.json")).write_text(json.dumps(
        {"last": last, "per_size": per_size, "dead_channels": dead, "hist": hist,
         "n_states_train": int(tr.sum()), "n_states_val": int(va.sum()),
         "n_routes": int(len(np.unique(route)))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{out} · {round(time.perf_counter() - t0)}s")
    print(f"최종 — MAE_pipe {last['MAE_pipe']:.2f} · MAE_sup {last['MAE_sup']:.2f} · "
          f"MAE_total {last['MAE_total']:.2f} (h 단독 {last['MAE_total_h_only']:.2f}) · "
          f"순위 일치 {last['rank_agree']:.3f} (h 단독 {last['rank_agree_h_only']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
