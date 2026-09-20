"""V-6: 스펙 5번(PPO 업데이트) 검증 — 정책경사가 h_net/τ 에 실제로 도달하는가."""
import io, contextlib, sys, importlib.util, numpy as np, torch
sys.argv=["x"]
spec=importlib.util.spec_from_file_location("ab","run_ablation.py"); ab=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
cells=ab.load_cells(ab.NB); ns={"__name__":"__main__"}
for tag,src in cells:
    if tag not in ab.BASE_CELLS: continue
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src.replace("USE_WANDB = True","USE_WANDB = False"),f"<{tag}>","exec"),ns)
    if tag=="2.": ns["cfg"].gnarl_mode=True
g=ns
env=g["PipeRoutingEnvStep1"](g["EVAL_POOL"], eval_mode=True)
obs,_=env.reset(options={"index":0}); m=env.action_masks()
pol=g["TrueGNARLPolicy"](env.observation_space, env.action_space, lambda _:3e-4)

# 상태 1개가 아니라 여러 에피소드에서 모은 배치로 본다.
# (클리핑이 걸린 상태에서는 torch.minimum 이 h_core 쪽 기울기를 0 으로 만든다)
OB=[]; MK=[]
for ep in range(6):
    o,_=env.reset(options={"index":ep})
    for t in range(12):
        mm=env.action_masks(); OB.append(o); MK.append(mm)
        o,r,te,tr,_=env.step(int(np.random.default_rng(t+ep).choice(np.where(mm)[0])))
        if te or tr: break
ot=torch.as_tensor(np.stack(OB)); mt=np.stack(MK)
d=pol.get_distribution(ot, mt)
a=torch.tensor([int(np.where(x)[0][0]) for x in MK])
logp=d.log_prob(a).sum()
logp.backward()
gh=sum(float(p.grad.abs().sum()) for p in pol.h_core.parameters() if p.grad is not None)
gt=float(pol.log_tau.grad.abs().sum()) if pol.log_tau.grad is not None else 0.0
ga=sum(float(p.grad.abs().sum()) for p in pol.action_net.parameters() if p.grad is not None)
gm=sum(float(p.grad.abs().sum()) for p in pol.mlp_extractor.parameters() if p.grad is not None)
with torch.no_grad():
    nb=ot[:,g["OBS_DIM"]:g["OBS_DIM"]+27*g["OBS_DIM"]].reshape(-1,g["OBS_DIM"])
    clip=float((torch.nn.functional.softplus(pol.h_core(nb))>pol.h_bound(nb)).float().mean())
print(f"V-6a log_prob 역전파 ({len(OB)}상태) → h_core grad {gh:.4e} | log_tau grad {gt:.4e} | "
      f"(미사용) action_net {ga:.1e} mlp_extractor {gm:.1e}")
print(f"     맨해튼 클리핑 발동률 {clip:.1%}  ← 이 비율만큼 h_core 로 가는 기울기가 0 이 된다")
# 클리핑을 끄면 기울기가 얼마나 커지는가
pol.zero_grad(); _cm=g["cfg"].h_clip_mode; g["cfg"].h_clip_mode="none"
pol.get_distribution(ot, mt).log_prob(a).sum().backward()
gh_noclip=sum(float(p.grad.abs().sum()) for p in pol.h_core.parameters() if p.grad is not None)
g["cfg"].h_clip_mode=_cm; pol.zero_grad()
pol.get_distribution(ot, mt).log_prob(a).sum().backward()
print(f"     클리핑 OFF 시 h_core grad {gh_noclip:.4e} (ON 대비 {gh_noclip/max(gh,1e-12):.1f}배)")

# value 경로는 v_core 로만
pol.zero_grad()
pol.predict_values(ot).sum().backward()
gv=sum(float(p.grad.abs().sum()) for p in pol.v_core.parameters() if p.grad is not None)
gh2=sum(float(p.grad.abs().sum()) for p in pol.h_core.parameters() if p.grad is not None)
print(f"V-6b value 역전파 → v_core grad {gv:.4e} | h_core grad {gh2:.1e} (분리돼야 0)")

# optimizer 에 h_core/log_tau 가 실제로 들어있는가
ids={id(p) for grp in pol.optimizer.param_groups for p in grp["params"]}
inopt=all(id(p) in ids for p in list(pol.h_core.parameters())+[pol.log_tau]+list(pol.v_core.parameters()))
print(f"V-6c optimizer 에 h_core+log_tau+v_core 포함: {inopt}")

# 실제 PPO 업데이트로 h_core 가 변하는가 (4096 스텝 = 1 rollout)
import copy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
def mk(r):
    def _i():
        e=g["PipeRoutingEnvStep1"](g["TRAIN_POOL"], eval_mode=False, seed=r)
        return Monitor(ActionMasker(e, lambda ee: ee.action_masks()), info_keywords=g["INFO_KEYS"])
    return _i
venv=DummyVecEnv([mk(i) for i in range(4)])
mdl=MaskablePPO(g["TrueGNARLPolicy"], venv, n_steps=128, batch_size=128, n_epochs=2,
                gamma=g["cfg"].gamma, seed=0, verbose=0, policy_kwargs=dict(net_arch=dict(pi=[64],vf=[64])))
before=copy.deepcopy(mdl.policy.h_core.state_dict()); tau0=float(mdl.policy.log_tau.exp())
mdl.learn(total_timesteps=1024, progress_bar=False)
after=mdl.policy.h_core.state_dict()
delta=sum(float((after[k]-before[k]).abs().sum()) for k in before)
print(f"V-6d 1024스텝 PPO 후 h_core 가중치 변화량 {delta:.4e} | τ {tau0:.4f} → {float(mdl.policy.log_tau.exp()):.4f}")
assert gt>0 and ga==0 and gm==0, "τ 미학습 또는 미사용 헤드가 학습된다"
assert gv>0 and gh2==0, "value 경로가 분리되지 않았다"
assert inopt and delta>0, "optimizer 누락 또는 h_core 가 학습되지 않는다"
print("\nV-6 통과 — 스펙 5번(PPO 가 h_net 을 직접 학습)이 실제로 성립한다")
