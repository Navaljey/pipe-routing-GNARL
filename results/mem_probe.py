#!/usr/bin/env python3
"""§4 메모리 예산 실측 — 학습 런 1개(현재 기본 설정 = H3fix)의 RSS 를 1초 간격으로 기록하고 최대 RSS 를 잰다.

run_step1_ab.py --child 를 그대로 실행한다 (노트북 셀 exec — 학습 로직을 따로 갖지 않는다).
사용: python3 mem_probe.py --steps 60000 --workdir <빈 디렉터리> --out ../results/step1_mem_probe.json
"""
import argparse, json, os, resource, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=60_000)
ap.add_argument("--workdir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cache = os.path.join(HERE, "..", "_ab", "_cache")
os.makedirs(a.workdir, exist_ok=True)
for f in os.listdir(cache):
    if f.startswith(("g3_scen_", "step1_rank_")) and not os.path.exists(os.path.join(a.workdir, f)):
        os.symlink(os.path.abspath(os.path.join(cache, f)), os.path.join(a.workdir, f))
cmd = [sys.executable, os.path.join(HERE, "run_step1_ab.py"), "--child", "--arm", "L", "--seed", "1",
       "--steps", str(a.steps), "--eval-every", "50000", "--ckpt-every", "500000"]
lg = open(os.path.join(a.workdir, "run.log"), "w")
t0 = time.time(); p = subprocess.Popen(cmd, cwd=a.workdir, stdout=lg, stderr=subprocess.STDOUT)
trace = []
while p.poll() is None:
    try:
        rss = int([l for l in open(f"/proc/{p.pid}/status") if l.startswith("VmRSS")][0].split()[1]) / 1024
        trace.append((round(time.time() - t0, 1), round(rss, 1)))
    except Exception:
        pass
    time.sleep(1)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
out = dict(steps=a.steps, rc=p.returncode, wall_s=round(time.time() - t0, 1), peak_rss_mb=round(peak, 1),
           trace_mb=trace)
json.dump(out, open(a.out, "w"), indent=1)
print(f"rc={p.returncode} · 최대 RSS {peak:.0f} MB · {out['wall_s']:.0f}s")
