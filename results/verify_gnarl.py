"""진짜 GNARL 구현 검증 V-1 ~ V-5."""
import io, contextlib, sys, importlib.util, numpy as np, torch
sys.argv=["x"]
spec=importlib.util.spec_from_file_location("ab","run_ablation.py"); ab=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)
cells=ab.load_cells(ab.NB)
ns={"__name__":"__main__"}
logs={}
for tag,src in cells:
    if tag not in ab.BASE_CELLS: continue
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src.replace("USE_WANDB = True","USE_WANDB = False"),f"<{tag}>","exec"),ns)
    if tag=="2.": ns["cfg"].gnarl_mode=True
    logs[tag]=buf.getvalue()
g=ns; cfg=g["cfg"]; R={}

# V-1 §11 환경 체크리스트 10/10
res=g["run_checklist_tests"]()
R["V-1 §11 체크리스트 10/10 (GNARL 모드)"]=(all(res.values()), f"{sum(res.values())}/{len(res)} 통과")

# 환경/정책 준비
env=g["PipeRoutingEnvStep1"](g["EVAL_POOL"], eval_mode=True)
obs,_=env.reset(options={"index":0})
R["V-0 관측 차원"]=(obs.shape[0]==g["GNARL_OBS_DIM"], f"{obs.shape[0]} == {g['GNARL_OBS_DIM']} (자기 {g['OBS_DIM']} + 이웃 27×{g['OBS_DIM']} + ΔJ 27)")

# V-2 이웃 인코딩 정합성: 관측에 실린 이웃 obs == 실제 그 상태의 build_obs
ok=True; bad=None
for i in range(26):
    d=g["DIRS26"][i]
    npos=(env.pos[0]+d[0], env.pos[1]+d[1], env.pos[2]+d[2])
    ns_=min(env.s+1,env.k) if (i==env.dir and env.k>0) else (1 if env.k>0 else 0)
    dJ=g["move_J_all"](env.dir)[i]
    ref=g["build_obs_batch"](env.pad, np.array([npos]), [i], [ns_], [env.g_J+dJ], env.goal, env.k)[0]
    got=obs[g["OBS_DIM"]+i*g["OBS_DIM"]:g["OBS_DIM"]+(i+1)*g["OBS_DIM"]]
    if not np.allclose(ref,got,atol=1e-6): ok=False; bad=i; break
# build_obs_batch 가 build_obs 와 일치하는지도 확인 (배치화 버그 방지)
ref1=g["build_obs"](env.pad, env.pos, env.dir, env.s, env.g_J, env.goal, env.k)
ref2=g["build_obs_batch"](env.pad, np.array([env.pos]), [env.dir], [env.s], [env.g_J], env.goal, env.k)[0]
R["V-2 이웃 인코딩 = 실제 이웃 상태의 obs"]=(ok and np.allclose(ref1,ref2,atol=1e-6),
    "26방향 전부 일치 + build_obs_batch==build_obs" if ok else f"방향 {bad} 불일치")

# 정책 생성 (무작위 h_net)
from gymnasium import spaces
pol=g["TrueGNARLPolicy"](env.observation_space, env.action_space, lambda _:3e-4)
ob_t=torch.as_tensor(obs[None])

# V-3 ΔJ 정합성: 관측의 ΔJ == move_J() 개별 호출
dJ_obs=obs[-27:]*g["J_MAX"]
dJ_ref=np.array([g["move_J"](i, env.dir, g["SPEC"])[0] for i in range(26)]+[0.0])
R["V-3 ΔJ 벡터화 = move_J() 개별 계산"]=(np.allclose(dJ_obs,dJ_ref,atol=1e-5),
    f"최대 오차 {np.max(np.abs(dJ_obs-dJ_ref)):.2e} kg")

# V-4 admissibility: h <= 맨해튼 상한 (전 이웃)
with torch.no_grad():
    nbr=ob_t[:,g["OBS_DIM"]:g["OBS_DIM"]+27*g["OBS_DIM"]].reshape(27,g["OBS_DIM"])
    h=pol.h_admissible(nbr); hb=pol.h_bound(nbr)
R["V-4 admissibility 클리핑 (h ≤ 맨해튼)"]=(bool((h<=hb+1e-6).all()),
    f"27개 전부 만족 | 클리핑 발동 {float((F_:=(torch.nn.functional.softplus(pol.h_core(nbr))>hb).float()).mean()):.0%}")

# V-5 마스크 준수 + 결정적 선택 = argmin(f) : 무작위 h_net 으로 200스텝
viol=0; argmin_ok=True; steps=0
for ep in range(8):
    o,_=env.reset(options={"index":ep})
    for t in range(40):
        m=env.action_masks()
        ot=torch.as_tensor(o[None])
        with torch.no_grad():
            a=int(pol._predict(ot, deterministic=True, action_masks=m[None])[0])
            f=pol.f_values(ot)[0].numpy()
        if not m[a]: viol+=1
        fm=np.where(m, f, np.inf)
        if a!=int(np.argmin(fm)): argmin_ok=False
        o,r,te,tr,_=env.step(a); steps+=1
        if te or tr: break
R["V-5 마스크 준수 + 평가시 argmin(f) 선택"]=(viol==0 and argmin_ok,
    f"{steps}스텝 중 마스크 위반 {viol}건, argmin 일치 {'전부' if argmin_ok else '불일치 있음'}")

print("="*88)
print("단계 1 — 구현 검증 V-1 ~ V-5")
print("="*88)
for k,(ok_,note) in R.items():
    print(f"  {'PASS' if ok_ else 'FAIL'}  {k}\n        → {note}")
print("\n[τ 근거 출력]"); print(logs.get("10b.","").strip())
assert all(v[0] for v in R.values()), "검증 실패"
print("\n단계 1 전부 통과")
