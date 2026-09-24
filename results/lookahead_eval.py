#!/usr/bin/env python3
"""S1-L2 단계 1 (§26) — 평가 전용 룩어헤드. 학습된 h 는 그대로 두고 **행동 선택만** 깊게 한다.

arm (§26-1):
  1ply   — 정책 그대로 (`argmax logits` = `argmin f`). 대조군.
  2ply   — argmin_a ΔJ_a + min_a' [ΔJ_a' + h(s'')]                  (칸 단위)
  beam4  — 깊이 3, 층마다 g_path + h 상위 4개만 전개                  (칸 단위)
  macro2 — 강제 직진 구간(유효 이동 1개)을 건너뛴 결정 지점 단위 2수

전이 모델은 노트북의 action_mask_27 · move_J_all · s 갱신 규칙을 그대로 쓴다 — A* 는 부르지 않는다 (§1).
목표 노드는 잎(h=0), 유효 이동 0개 노드는 잎(h + FAIL). 1-ply 는 정책을 직접 쓰고, 같은 스텝에서
이 스크립트의 h 기반 1-ply 도 계산해 **불일치 수**를 센다 (건전성 관문, §26-1).

사용 (캐시가 있는 _ab/_cache 에서):
  python3 lookahead_eval.py ../L_300k_h3_s1 --out ../../results/_la/s1.json [--methods 1ply,2ply,beam4,macro2]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from regrade import build_ns          # noqa: E402  (노트북 셀 3~9 정의부를 그대로 exec)
from run_step1_ab import ARMS         # noqa: E402

APPROACH_ZONE = 8
FAIL = 10.0                           # 데드락 잎 비용 (정규화 J 단위 — 경로 J 는 0.x 수준)


class Planner:
    def __init__(self, ns, policy, pad, goal, k):
        self.ns, self.policy, self.pad, self.k = ns, policy, pad, k
        self.goal = tuple(int(v) for v in goal)
        self.goal_a = np.asarray(self.goal, dtype=np.int64)
        self.JM = float(ns["J_MAX"])
        self.DIRS = ns["DIRS_ARR"]
        self.n_h = 0                  # h 평가 상태 수 (비용 계측)

    # ---- 전이 모델 (환경 step 과 같은 규칙) ----
    def children(self, node):
        pos, d, s, g = node
        m = self.ns["action_mask_27"](self.pad, pos, d, s, self.goal, self.k)
        acts = np.where(m[:26])[0]
        dJ = self.ns["move_J_all"](d)
        out = []
        for a in acts:
            a = int(a)
            npos = (pos[0] + int(self.DIRS[a, 0]), pos[1] + int(self.DIRS[a, 1]), pos[2] + int(self.DIRS[a, 2]))
            ns_ = min(s + 1, self.k) if a == d else 1
            out.append((a, (npos, a, ns_, g + float(dJ[a])), float(dJ[a]) / self.JM))
        return out

    def h(self, nodes):
        if not nodes:
            return np.zeros(0)
        P = np.array([n[0] for n in nodes], dtype=np.int64)
        D = np.array([n[1] for n in nodes], dtype=np.int64)
        S = np.array([n[2] for n in nodes], dtype=np.int64)
        G = np.array([n[3] for n in nodes], dtype=np.float64)
        obs = self.ns["build_obs_batch"](self.pad, P, D, S, G, self.goal, self.k)
        self.n_h += len(nodes)
        with torch.no_grad():
            return self.policy.h_admissible(torch.as_tensor(obs)).numpy()[:, 0].astype(np.float64)

    def is_goal(self, node):
        return node[0] == self.goal

    # ---- 1-ply (정책 규칙: 목표 이웃에도 h(obs) 를 쓴다) — 건전성 확인용 ----
    def one_ply(self, root):
        ch = self.children(root)
        if not ch:
            return None
        hv = self.h([c[1] for c in ch])
        sc = np.array([c[2] for c in ch]) + hv
        return ch[int(np.argmin(sc))][0]

    # ---- 칸 단위 2-ply ----
    def two_ply(self, root):
        ch = self.children(root)
        if not ch:
            return None
        vals = np.full(len(ch), np.inf)
        leaf_nodes, leaf_owner, leaf_cost = [], [], []
        dead_nodes, dead_owner = [], []
        for i, (a, c, dj) in enumerate(ch):
            if self.is_goal(c):
                vals[i] = dj
                continue
            gc = self.children(c)
            if not gc:
                dead_nodes.append(c); dead_owner.append(i); continue
            for (a2, c2, dj2) in gc:
                if self.is_goal(c2):
                    vals[i] = min(vals[i], dj + dj2)
                else:
                    leaf_nodes.append(c2); leaf_owner.append(i); leaf_cost.append(dj + dj2)
        if leaf_nodes:
            hv = self.h(leaf_nodes)
            for o, cst, hh in zip(leaf_owner, leaf_cost, hv):
                vals[o] = min(vals[o], cst + hh)
        if dead_nodes:
            hv = self.h(dead_nodes)
            for o, hh in zip(dead_owner, hv):
                vals[o] = min(vals[o], ch[o][2] + hh + FAIL)
        return ch[int(np.argmin(vals))][0]

    # ---- 칸 단위 beam (폭 B, 깊이 D) ----
    def beam(self, root, B=4, D=3):
        frontier = [(0.0, None, root)]            # (g_path, 첫 수, 노드)
        done = []                                 # (f, 첫 수)
        for depth in range(D):
            kids = []
            dead = []
            for gp, first, node in frontier:
                ch = self.children(node)
                if not ch:
                    dead.append((gp, first, node)); continue
                for a, c, dj in ch:
                    fa = a if first is None else first
                    if self.is_goal(c):
                        done.append((gp + dj, fa))
                    else:
                        kids.append((gp + dj, fa, c))
            if dead:
                hv = self.h([n for _, _, n in dead])
                done += [(gp + hh + FAIL, fa) for (gp, fa, _), hh in zip(dead, hv)]
            if not kids:
                frontier = []
                break
            hv = self.h([c for _, _, c in kids])
            f = np.array([gp for gp, _, _ in kids]) + hv
            order = np.argsort(f, kind="stable")[:B]
            frontier = [kids[j] for j in order]
            fvals = {j: f[j] for j in order}
            if depth == D - 1:
                done += [(fvals[j], kids[j][1]) for j in order]
        if not done:
            return None
        return min(done, key=lambda x: x[0])[1]

    # ---- 결정 지점 단위 2-ply ----
    def advance(self, node, a_child, cap):
        """a_child = (a, 노드, dj). 강제 구간(유효 이동 1개)을 끝까지 따라간다 → (노드, 누적 dj, 상태)."""
        _, c, cost = a_child
        for _ in range(cap):
            if self.is_goal(c):
                return c, cost, "goal"
            ch = self.children(c)
            if not ch:
                return c, cost, "dead"
            if len(ch) >= 2:
                return c, cost, "decide"
            _, c, dj = ch[0]
            cost += dj
        return c, cost, ("goal" if self.is_goal(c) else "decide")

    def macro2(self, root):
        cap = 3 * (self.k + 1)
        ch = self.children(root)
        if not ch:
            return None
        vals = np.full(len(ch), np.inf)
        leaves, owner, costs, pen = [], [], [], []
        for i, ac in enumerate(ch):
            n1, c1, st1 = self.advance(root, ac, cap)
            if st1 == "goal":
                vals[i] = c1; continue
            if st1 == "dead":
                leaves.append(n1); owner.append(i); costs.append(c1); pen.append(FAIL); continue
            for ac2 in self.children(n1):
                n2, c2, st2 = self.advance(n1, ac2, cap)
                if st2 == "goal":
                    vals[i] = min(vals[i], c1 + c2)
                else:
                    leaves.append(n2); owner.append(i); costs.append(c1 + c2)
                    pen.append(FAIL if st2 == "dead" else 0.0)
        if leaves:
            hv = self.h(leaves)
            for o, cst, p_, hh in zip(owner, costs, pen, hv):
                vals[o] = min(vals[o], cst + hh + p_)
        return ch[int(np.argmin(vals))][0]


def split_segments(dists, gJ, turns, base, goal):
    """§24-6 과 같은 규칙: 목표 맨해튼 8칸 구역 첫 진입 시점에서 자른다 (에이전트·A* 동일)."""
    d = np.asarray(dists); inz = d <= APPROACH_ZONE
    ent = int(np.argmax(inz)) if inz.any() else len(d) - 1
    bst = base["states"]
    bd = np.array([int(np.abs(np.asarray(p_, dtype=int) - goal).sum()) for (p_, _, _, _) in bst])
    bent = int(np.argmax(bd <= APPROACH_ZONE)) if (bd <= APPROACH_ZONE).any() else 0
    b_g = np.array([g_ for (_, _, _, g_) in bst])
    b_turn = np.array([int(b[1] != a_[1]) for a_, b in zip(bst[:-1], bst[1:])])
    turns = np.asarray(turns)
    return dict(
        J_cruise=float(gJ[ent]), J_approach=float(gJ[-1] - gJ[ent]),
        J_cruise_base=float(b_g[bent]), J_approach_base=float(base["J"] - b_g[bent]),
        elb_cruise=int(turns[:ent].sum()), elb_approach=int(turns[ent:].sum()),
        elb_cruise_base=int(b_turn[:bent].sum()), elb_approach_base=int(b_turn[bent:].sum()),
        steps_approach=int(len(d) - 1 - ent), steps_approach_base=int(len(bst) - 1 - bent),
        reentries=int(np.sum(inz[1:] & ~inz[:-1])),
    )


def run_method(ns, model, method, sanity=True):
    env = ns["PipeRoutingEnvStep1"](ns["EVAL_POOL"], eval_mode=True)
    STAY = ns["STAY"]
    rows, t_choose, n_steps, n_h, mism, n_cmp = [], 0.0, 0, 0, 0, 0
    t_wall0 = time.perf_counter()
    for i in range(len(ns["EVAL_POOL"])):
        obs, _ = env.reset(options={"index": i})
        goal = np.asarray(env.goal, dtype=np.int64)
        pl = Planner(ns, model.policy, env.pad, env.goal, env.k)
        dists = [int(np.abs(np.asarray(env.pos) - goal).sum())]
        gJ = [0.0]; turns = []
        done = False; info = {}
        while not done:
            root = (tuple(int(v) for v in env.pos), int(env.dir), int(env.s), float(env.g_J))
            m = env.action_masks()
            t0 = time.perf_counter()
            if method == "1ply":
                with torch.no_grad():
                    dist = model.policy.get_distribution(torch.as_tensor(obs[None]), action_masks=m[None])
                    a = int(dist.get_actions(deterministic=True)[0])
            elif method == "2ply":
                a = pl.two_ply(root)
            elif method == "beam4":
                a = pl.beam(root, B=4, D=3)
            elif method == "macro2":
                a = pl.macro2(root)
            else:
                raise ValueError(method)
            t_choose += time.perf_counter() - t0
            if method == "1ply" and sanity:                # 건전성: h 기반 1-ply 와 정책의 행동이 같은가
                mine = pl.one_ply(root)
                if mine is not None:
                    n_cmp += 1; mism += int(mine != a)
            if a is None:
                a = STAY
            assert m[a], "마스크 밖 행동"
            turns.append(int(a != STAY and a != env.dir))
            obs, r, term, trunc, info = env.step(a)
            n_steps += 1
            dists.append(int(np.abs(np.asarray(env.pos) - goal).sum())); gJ.append(float(env.g_J))
            done = term or trunc
        n_h += pl.n_h
        row = dict(idx=i, **{k: info[k] for k in ("success", "deadlock", "timeout", "j_ratio", "elbow_ratio",
                                                   "n_bends", "bends_base", "ep_steps", "j_agent")})
        if env.base is not None:
            row.update(split_segments(dists, np.asarray(gJ), turns, env.base, goal))
            row["j_base"] = float(env.base["J"])
        rows.append(row)
    wall = time.perf_counter() - t_wall0
    return rows, dict(wall_s=wall, choose_s=t_choose, n_steps=n_steps,
                      ms_per_step=1000 * t_choose / max(n_steps, 1),
                      h_states=n_h, h_per_step=n_h / max(n_steps, 1),
                      policy_vs_h1ply_mismatch=mism, policy_vs_h1ply_compared=n_cmp)


def aggregate(rows):
    S = [r for r in rows if r["success"] > 0.5 and "J_cruise" in r]
    f = lambda k: float(np.mean([r[k] for r in S])) if S else float("nan")
    sm = lambda k: float(np.sum([r[k] for r in S]))
    er = [r["elbow_ratio"] for r in S if r["elbow_ratio"] == r["elbow_ratio"]]
    return dict(
        success_rate=float(np.mean([r["success"] for r in rows])),
        deadlock_rate=float(np.mean([r["deadlock"] for r in rows])),
        n_success=len(S),
        j_ratio=f("j_ratio"),
        elbow_ratio=float(np.mean(er)) if er else float("nan"),
        jr_cruise=sm("J_cruise") / max(sm("J_cruise_base"), 1e-9),
        jr_approach=sm("J_approach") / max(sm("J_approach_base"), 1e-9),
        exc_total=float(np.mean([r["j_agent"] - r["j_base"] for r in S])) if S else float("nan"),
        exc_cruise=float(np.mean([r["J_cruise"] - r["J_cruise_base"] for r in S])) if S else float("nan"),
        exc_approach=float(np.mean([r["J_approach"] - r["J_approach_base"] for r in S])) if S else float("nan"),
        elb_cruise=f("elb_cruise"), elb_approach=f("elb_approach"),
        elb_cruise_base=f("elb_cruise_base"), elb_approach_base=f("elb_approach_base"),
        steps_approach=f("steps_approach"), steps_approach_base=f("steps_approach_base"),
        reentries=f("reentries"), ep_steps=float(np.mean([r["ep_steps"] for r in rows])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="1ply,2ply,beam4,macro2")
    ap.add_argument("--ckpt", default="checkpoint_step1_final")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-sanity", action="store_true", help="시간 측정용 — 1-ply 건전성 비교를 끈다")
    a = ap.parse_args()
    torch.set_num_threads(1)
    override = dict(ARMS["L"]); override.update(h_rank_batches=256, obs_align_norm="k", obs_capture=True)
    for kv in a.set:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except Exception: pass
        override[k] = v
    from sb3_contrib import MaskablePPO
    ns = build_ns(override)
    model = MaskablePPO.load(os.path.join(a.run_dir, a.ckpt + ".zip"), device="cpu", print_system_info=False)
    model.policy.eval()
    out = dict(run=os.path.basename(a.run_dir.rstrip("/")), timesteps=int(model.num_timesteps), methods={})
    for meth in a.methods.split(","):
        rows, tm = run_method(ns, model, meth, sanity=not a.no_sanity)
        agg = aggregate(rows); agg.update(tm)
        out["methods"][meth] = dict(agg=agg, rows=rows)
        print(f"{out['run']} {meth:7s} succ {agg['success_rate']:.3f} jr {agg['j_ratio']:.3f} "
              f"(cruise {agg['jr_cruise']:.3f} / appr {agg['jr_approach']:.3f}) exc {agg['exc_total']:.1f} "
              f"[c {agg['exc_cruise']:.1f} / a {agg['exc_approach']:.1f}] elb {agg['elbow_ratio']:.2f} "
              f"appr_steps {agg['steps_approach']:.1f} | {tm['ms_per_step']:.2f} ms/step, wall {tm['wall_s']:.0f}s"
              + (f" | mismatch {tm['policy_vs_h1ply_mismatch']}/{tm['policy_vs_h1ply_compared']}"
                 if meth == "1ply" else ""), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
