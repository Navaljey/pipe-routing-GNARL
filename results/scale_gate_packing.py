#!/usr/bin/env python3
"""§42 ② 관문 — 시나리오 비트 패킹: 검증 격자 동일성 · 롤아웃 동일 · pickle 왕복 · 실규모 메모리/시간 (_ab/_cache 에서 실행)."""
import sys, io, contextlib, pickle, time, json, numpy as np
sys.path.insert(0, "/home/user/pipe-routing-GNARL/results")
from run_step1_ab import load_cells, NB
CELLS = load_cells(NB)
def boot(pack=None, dims=None, upto=("3.","4.","5.")):
    ns = {"__name__": "__main__", "USE_WANDB": False, "run": None}
    exec(compile("import os, sys, time", "<p>", "exec"), ns)
    for tag, src in CELLS:
        if tag in upto:
            if tag == "3." and dims:
                src = src.replace("cfg = Step1Config()", f"cfg = Step1Config(NX={dims[0]}, NY={dims[1]}, NZ={dims[2]})")
            if tag == "4." and dims:
                src = src[:src.find("CACHE = (")]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, f"<nb:{tag}>", "exec"), ns)
            if tag == "3.":
                ns["cfg"].pretrain_h=False; ns["cfg"].bc_batches=0; ns["cfg"].pack_scenarios = pack
    return ns
out = {}
A = boot(None); B = boot(True)
out["default_pack_flag"] = bool(A["PACK_SCENARIOS"]); out["forced_pack_flag"] = bool(B["PACK_SCENARIOS"])
bad = 0
for a, b in zip(A["TRAIN_POOL"] + A["EVAL_POOL"], B["TRAIN_POOL"] + B["EVAL_POOL"]):
    bad += not (np.array_equal(a["free"], b["free"]) and np.array_equal(a["pad"], b["pad"]) and "free" not in dict.keys(b) and "pad" not in dict.keys(b))
out["verify_grid_mismatch"] = bad
out["verify_bits_MB"] = sum(sc["free_bits"].nbytes for sc in B["TRAIN_POOL"] + B["EVAL_POOL"]) / 1e6
out["verify_bool_MB"] = sum(sc["free"].nbytes + sc["pad"].nbytes for sc in A["TRAIN_POOL"] + A["EVAL_POOL"]) / 1e6
# env rollouts identical (random actions, fixed seed)
def roll(ns):
    env = ns["PipeRoutingEnvStep1"](ns["EVAL_POOL"], eval_mode=True); rng = np.random.default_rng(0); log = []
    for i in range(len(ns["EVAL_POOL"])):
        o, _ = env.reset(options={"index": i})
        for _ in range(120):
            m = env.action_masks(); a = int(rng.choice(np.where(m)[0]))
            o, r, te, tr, _ = env.step(a); log.append((float(o.sum()), float(r), te, tr))
            if te or tr: break
    return log
out["rollout_identical"] = roll(A) == roll(B)
# pickle roundtrip of plain pool
blob = pickle.dumps(B["plain_pool"](B["EVAL_POOL"])); back = B["prepare_pool"](pickle.loads(blob))
out["pickle_roundtrip_ok"] = all(np.array_equal(x["pad"], y["pad"]) for x, y in zip(back, A["EVAL_POOL"]))
# full scale: generation fill/time, packed size, unpack time
F = boot(None, dims=(400, 300, 100), upto=("3.", "4."))
out["full_auto_pack"] = bool(F["PACK_SCENARIOS"])
rng = np.random.default_rng(0); gens = []
for _ in range(3):
    t0 = time.time(); free = F["gen_obstacle_grid"](rng); tg = time.time() - t0
    sc = F["pack_scenario"](dict(free=free, pad=F["make_pad"](free)))
    F["_UNPACK_CACHE"].clear(); t0 = time.time(); p = sc["pad"]; tu = time.time() - t0
    t0 = time.time(); p2 = sc["pad"]; th = time.time() - t0
    gens.append(dict(gen_sec=round(tg, 2), fill=float((~free).mean()), bits_MB=sc["free_bits"].nbytes / 1e6,
                     bool_MB=(free.nbytes + F["make_pad"](free).nbytes) / 1e6, unpack_pad_sec=round(tu, 3), cache_hit_sec=round(th, 6),
                     pad_ok=bool(np.array_equal(p, F["make_pad"](free)))))
out["full"] = gens
json.dump(out, open("gate2.json", "w"), indent=1); print(json.dumps(out, indent=1))
