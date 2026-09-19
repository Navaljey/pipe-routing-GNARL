# CLAUDE.md — 파이프 라우팅 RL v3
**작성일:** 2026-09-19  
**상태:** 초안 (V1~V3, G1~G2 검증 완료 기준)

---

## 0. 이 문서의 목적과 규칙

이 문서는 실측 검증 결과에만 기반한다. 추측은 "추측:" 태그로, 미결 사항은 "미결:" 태그로 명시한다. 검증 없이 확정으로 적힌 문장은 없다.

### 변경 불가 항목 (실측으로 이미 확정)

변경하려면 해당 검증을 재실행하고 결과를 기록해야 한다.

- RL 알고리즘 본체: **sb3-contrib MaskablePPO 고정** (G2 통과)
- 그래프 상태 에피소드당 재구성: **금지** (프로그램 3 실패 원인)
- 자체 PPO 구현: **금지** (프로그램 2 실패 원인)
- 벤딩 반경 처리: **action mask (보상 항목 금지)** (V1 통과)
- 목적함수 J: **규격/단가표 상수 — 탐색 대상 아님** (V2 확정)

---

## 1. 프로젝트 목표

숙련 엔지니어가 미처 제시하지 못하는 파이프 라우팅을 AI가 제시한다.

- 숙련공 시연을 정답으로 사용하지 않는다 (시연이 성능 천장이 되기 때문)
- 기존 CAD 설계 데이터를 학습 데이터로 사용하지 않는다 (휴먼 에러 답습 방지)
- 보상 신호만으로 학습 (RL)

---

## 2. 아키텍처 — 3층 구조

3층은 검증으로 확정된 설계이며, 같은 층의 항목을 혼합하지 않는다.

```
Layer 0 │ 하드 제약 (Hard Constraints)
        │ → action_masks() 로만 처리. 보상에 절대 포함하지 않는다.
        │ → 제약 위반 = "선택 불가" (벌점이 아님)
        │
Layer 1 │ 목적함수 J (단일 물리 단위: kg)
        │ → 규격/단가표에서 조회한 상수. 탐색 대상 아님.
        │ → 모든 Step이 같은 J 위에서 "항 추가" 방식으로 확장
        │
Layer 2 │ 학습 신호 (PBS Shaping)
        │ → r = -ΔJ + [γΦ(s') - Φ(s)]
        │ → Φ(종단) = 0, γ_PBS = γ_PPO 강제 (최적정책 불변 조건)
        │ → 여기의 계수만 autoresearch 탐색 대상
```

### Layer 0 항목 (확정)

| 제약 | 처리 | 검증 |
|---|---|---|
| 충돌 (장애물, 기존 파이프) | mask | v2 기존 |
| 경계 위반 | mask | v2 기존 |
| 역방향 이동 | mask | V3-A |
| 벤딩 반경 (ASME B31.3) | mask + lookahead k | V1 통과 |
| 역구배 (중력관, Step 3) | mask | 미실측, 설계 확정 |
| 밸브 높이 범위 (Step 5) | mask | 미실측, 설계 확정 |
| 서포트 최대간격 (Step 6) | mask | 미실측, 설계 확정 |

### Layer 1 — 목적함수 J (단위: kg)

```
Step 1: J = Σ dist(칸) × CELL_M × kg_per_m
Step 2: J = J_step1 + n_bends_90 × elbow90_kg + n_bends_45 × elbow45_kg
Step 4: J = J_step2 + n_tee × tee_kg
Step 5: J = J_step4 + n_valve × valve_kg
Step 6: J = J_step5 + n_support × support_kg  ← support_kg 미확정
Step 7~: J = Σ J (N개 파이프 합산)
```

규칙: 이전 Step 항의 계수는 바뀌지 않는다. 항만 추가된다.

---

## 3. JIS 물리 상수 (V2 확정)

스케줄: **Sch40 전 구경 고정**  
밸브 타입: **게이트밸브만**  
용접/플랜지: **무시**  
서포트 실중량: **미확정 — Step 6 진입 시 현장 데이터로 보완**

### 직관 중량 (JIS G3454 Sch40)

| 호칭경 | OD (mm) | kg/m | 비고 |
|---|---|---|---|
| 15A | 21.7 | 1.27 | |
| 25A | 34.0 | 2.57 | CC 파이프 |
| 50A | 60.5 | 5.44 | |
| 65A | 76.3 | 9.11 | BA, BB 파이프 |
| 100A | 114.3 | 16.10 | |
| 150A | 165.2 | 28.20 | |
| 200A | 216.3 | 42.50 | |
| 300A | 318.5 | 79.70 | AA 파이프 |
| 400A | 406.4 | 124.00 | |
| 500A | 508.0 | 185.00 | |

### 관이음 중량 (JIS B2312, LR)

| 호칭경 | 90° 엘보 (kg) | 45° 엘보 (kg) | 티 (kg) | 게이트밸브 (kg) | B/L |
|---|---|---|---|---|---|
| 25A | 0.30 | 0.17 | 0.54 | 2.0 | 0.12m |
| 65A | 1.80 | 0.99 | 3.24 | 10.0 | 0.20m |
| 100A | 3.80 | 2.09 | 6.84 | 20.0 | 0.24m |
| 200A | 18.0 | 9.90 | 32.4 | 80.0 | 0.42m |
| 300A | 51.0 | 28.1 | 91.8 | 210.0 | 0.64m |
| 500A | 195. | 107. | 351. | 730.0 | 1.05m |

**B/L 전 구경 범위: [0.08, 1.05]m** — 이 범위가 autoresearch의 실제 탐색 의미 구간.

---

## 4. 행동 공간 (V3 확정)

### 27방향 (V3-A 검증)

```python
DIRS26 = [
    (dx,dy,dz)
    for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
    if not (dx==0 and dy==0 and dz==0)
]   # 26개

# 역방향: 벡터 부호 반전 (V3-A: 26개 모두 정확 확인)
DIR_TO_IDX = {d:i for i,d in enumerate(DIRS26)}
OPP26 = {i: DIR_TO_IDX[(-d[0],-d[1],-d[2])] for i,d in enumerate(DIRS26)}
```

행동 집합: **26방향 + STAY(1) = 27**

역방향 금지: **매 스텝 OPP26[현재방향] 1개 마스킹**

### 이동 거리 (V3-B 검증)

```python
MOVE_DIST = {
    d: math.sqrt(sum(x**2 for x in d))  # 면=1.0, 모서리=√2, 꼭짓점=√3
    for d in DIRS26
}

def step_cost(d, kg_per_m):
    return MOVE_DIST[d] * CELL_M * kg_per_m
```

---

## 5. 벤딩 반경 마스킹 (V1 확정)

### k값 (LR 90°, CELL=50mm 기준)

| 호칭경 | k (칸) | 필요 직진 (mm) |
|---|---|---|
| 15A~25A | 1~2 | 50~100 |
| 50A~65A | 3 | 150 |
| 100A | 6 | 300 |
| 150A | 9 | 450 |
| 200A | 12 | 600 |
| 300A | 18 | 900 |
| 500A | 30 | 1,500 |

### 마스킹 규칙

```python
def is_valid_turn(g, cur_pos, cur_dir, new_dir, s, k):
    # 1. 직진 칸수 조건
    if s < k:
        return False
    # 2. lookahead k: 새 방향으로 k칸이 비어있는가 (V1 핵심)
    dr,dc,dz = DIRS26[new_dir]
    nr,nc,nz = cur_pos[0]+dr, cur_pos[1]+dc, cur_pos[2]+dz
    for i in range(1, k+1):
        if not free3(g, nr+dr*i, nc+dc*i, nz+dz*i):
            return False
    return True
```

lookahead 없으면 도달가능 상태의 2%에서 데드락 발생 (V1 실측).

### 속도

on-demand 계산: k=30 기준 200만 스텝 누적 14초 (V1 실측).  
사전계산 금지: 3D 전체 run-length 사전계산 = 69MB (§4.3 24MB 초과).

### 시나리오 생성기 수정 필요

현재 생성기는 k=0 A*로 도달성 검증 → k 제약 하에서 도달 불가 시나리오가 섞임 (100A 이상 8%).  
→ **시나리오 생성 시 구경별 k 제약 하에서 도달성 재검증 필요.**

---

## 6. GNARL + MaskablePPO 통합 구조 (G1, G2 확정)

### G1 — J 기반 A* 구조

```python
def astar_gnarl(g, start, goal, J_fn, h_net, k):
    """
    g(n) = J_fn(이동칸수, 벤딩수)  ← 보상함수 변환 불필요
    h(n) = h_net(상태, 목표)       ← 신경망이 학습
    f(n) = g(n) + h(n)
    """
```

J를 g(n)에 직접 넣으면 보상함수 계수 설계가 불필요하다 (G1-Q1 통과).

### G2 — sb3-contrib 통합

```python
class GNARLFeaturesExtractor(BaseFeaturesExtractor):
    """
    sb3의 BaseFeaturesExtractor 상속.
    PPO 알고리즘 본체(rollout buffer, advantage, clip loss) 수정 없음.
    h(n) 신경망을 feature extractor 범위 안에 삽입.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.h_net = nn.Sequential(...)    # h(n) 추정
        self.feat_net = nn.Sequential(...) # 일반 특징
    
    def forward(self, obs):
        h_val = self.h_net(obs)            # 스칼라 1개
        feats = self.feat_net(obs)
        return torch.cat([feats, h_val], dim=-1)
```

G2 통과 사항: Feature Extractor 상속 ✅, rollout 수집 ✅, advantage+clip loss ✅, action mask 유지 ✅

### admissibility 조건 (G1 speed_compare에서 발견)

```python
# h(n)이 h*(n)을 초과하면 A*가 더 많은 노드 탐색
# 안전 클리핑:
def h_admissible(state, goal, shape):
    h_learned = net.h(state, goal, shape)
    h_manhattan = manhattan_distance(state[:3], goal) * CELL_M * kg_per_m
    return min(h_learned, h_manhattan)  # 맨해튼 초과 금지
```

### J 정규화 필수 (G1-Q2에서 발견)

```python
DIAG_J = math.sqrt(NX**2+NY**2+NZ**2) * CELL_M * kg_per_m
SCALE = 1.0 / DIAG_J   # 모든 J를 [0,1]로 정규화
```

정규화 없으면 gradient 불안정으로 h(n) 학습 실패 (G1 1차 시도 실패).

### Step 전이 (G1-Q3 확정)

J가 "항 추가" 구조이므로 Step N 신경망이 Step N+1 출발점이 된다.  
"길이를 줄이는 h값" = "엘보도 줄이는 방향" 이기 때문 (G1-Q3 8/8 통과).

---

## 7. Step별 로드맵

### Step 1~6: 단일 파이프

| Step | 추가 제약 | Layer 0 추가 | Layer 1 추가 항 |
|---|---|---|---|
| 1 | 기본 경로 | 충돌, 역주행 | 길이 (kg) |
| 2 | 벤딩 반경 | lookahead k 마스킹 | 엘보 (kg) |
| 3 | 중력관 구배 | 역구배 마스킹 | — (구배는 Layer 0) |
| 4 | 분기 (Tee) | 분기각도 제약 | 티 (kg) |
| 5 | 밸브 + 접근성 | 높이/여유 마스킹 | 밸브 (kg) |
| 6 | 서포트 최적화 | 최대간격 마스킹 | 서포트 (kg) ← 미확정 |

### Step 7~10: 다중 파이프

| Step | 파이프 수 | 핵심 과제 |
|---|---|---|
| 7 | 2 | 양보(yielding) 학습 |
| 8 | 8 (타입별) | 이질적 타입 동시 배치 |
| 9 | 50 | 순서 조합 |
| 10 | 300 | 대규모 동시 최적화 |

Step 7~10 아키텍처는 **L-D1b로 미결**. Step 6 종료 후 판단.

---

## 8. 관측 벡터 구조

```
obs[0:3]    현재 위치 xyz (정규화)
obs[3:6]    목표 위치 xyz (정규화)
obs[6:32]   현재 방향 one-hot (26방향)
obs[32]     직진 칸수 s (= min(s,k)/k 정규화)
obs[33]     g_cost 누적 (= g_cost / J_max 정규화)
obs[34:40]  이웃 6면 SDF (자유=+1, 막힘=-1)
obs[40:50]  SDF 센서 (§4.3 기존 유지)
...         Step별 추가 채널 (Step 2: +5, Step 3: +3, ...)
```

총 dim은 Step이 늘수록 증가. load_state_dict 호환을 위해 **추가 채널은 뒤에 붙이고 zero-padding으로 전이**.

---

## 9. PBS 쉐이핑 (probe3/4 확정)

### 조건 (위반 시 목적함수가 바뀜)

```
1. Φ(종단) = 0  ← 위반하면 최적정책 20/20 전부 변경 (probe4 실측)
2. γ_PBS = γ_PPO
3. Φ는 state-only 함수
```

### Φ 후보 비교 결과 (V4 완료)

| 후보 | 수식 | Φ(종단)=0 | ΔΦ 표준편차 | 비고 |
|---|---|:---:|---|---|
| A. 맨해튼 | -manhattan×CELL_M×kg/J_max | ✅ | **0.0646** | 신호 1위 |
| B. 유클리드 | -euclidean×CELL_M×kg/J_max | ✅ | 0.0369 | |
| C. 유클리드+혼잡도 | 0.8×B - 0.2×congestion | ✅ | 0.0294 | |
| D. 혼잡도만 | -1/(sdf×5+1) | ✅ | 0.0052 | 단독으로는 신호 약함 |
| E. 복합(B×0.7+D×0.3) | 0.7×B + 0.3×D | ✅ | 0.0259 | |

**잠정 선택: A. 맨해튼** — Φ(종단)=0 준수 + ΔΦ 표준편차 최대

```python
def Phi_goal_manhattan(pos, goal):
    if pos == goal: return 0.0   # 종단 조건 명시
    d = sum(abs(a-b) for a,b in zip(pos, goal))
    return -d * CELL_M * kg_per_m / J_MAX
```

**단, V4-3 수렴 속도 실측은 G3(Colab)에서 확정.** 소규모 격자는 비교 의미 없음.
autoresearch에서 맨해튼 vs 복합 비율 함께 탐색 가능.

---

## 10. autoresearch 탐색 범위 (V2+V3 확정)

### 탐색 대상

Layer 2 Φ 계수만 탐색한다.

```python
# 탐색 대상 (V2+V4 확정 이후)
search_space = {
    "phi_type":         ["manhattan", "combined"],  # V4-D: 잠정 맨해튼, G3에서 확정
    "phi_goal_ratio":   (0.6, 1.0),   # 복합일 때 goal 비중
    "phi_cong_ratio":   (0.0, 0.4),   # 복합일 때 혼잡도 비중 (1-goal_ratio)
    "learning_rate":    (1e-4, 5e-4),
    "n_steps":          (256, 2048),
}

# 탐색 대상 아님 (V2 확정 — 물리 상수)
fixed = {
    "pipe_kg_per_m":  (구경별 V2 표),
    "elbow90_kg":     (구경별 V2 표),
    "elbow45_kg":     (구경별 V2 표),
    "tee_kg":         (구경별 V2 표),
    "valve_kg":       (구경별 V2 표),
}
```

B/L 비율은 위 고정값에서 자동 결정됨. B/L sweep 불필요.

---

## 11. 환경 구현 시 필수 체크리스트

V2·V3·G2에서 발견한 버그 패턴.

```
[ ] step() 에 경계 검사 추가
    → 무효 행동 시 제자리 유지 + 페널티 (G2-D 버그 원인)

[ ] action_masks() 현재 위치 유효성 확인
    → 경계 밖 pos 이면 전부 False 반환

[ ] 초기 상태 방향을 하드코딩 금지
    → 모든 유효 첫이동 방향을 초기 상태 집합으로 (V3-D 버그 원인)

[ ] 시나리오 생성기: k 제약 하에서 도달성 검증
    → 기존 k=0 A* 도달성으로는 부족 (V1 발견)

[ ] J 정규화: DIAG_J = sqrt(NX²+NY²+NZ²) × CELL_M × kg_per_m
    → 미정규화 시 h(n) 학습 실패 (G1-Q2 1차 시도)

[ ] h(n) admissibility 클리핑
    → h_learned > h_manhattan 이면 맨해튼으로 클리핑 (speed_compare 발견)
```

---

## 12. 과거 프로그램 폐기 이력

폐기 사유가 "명시적 기각"인 것만 재활용 금지. "부수 삭제"는 재검토 가능.

| 항목 | 상태 | 사유 | 재활용 가능성 |
|---|---|---|---|
| 자체 구현 PPO | **금지** | advantage DoF≤0 버그 (프로그램 2 실측) | ❌ |
| 그래프 상태 매 step 재구성 | **금지** | O(노드×장애물) 비용 (프로그램 3 실측) | ❌ |
| 3D CNN 관측 | 기각 | OOM + 보상 희소성 (§4.3) | ❌ |
| 숙련공 시연 + IRL | 기각 | 시연이 성능 천장 (§1 철학) | ❌ |
| Dual Graph + Hetero GNN | 미검증 | 트레이너 버그로 학습 못 함 | 🔶 재검토 가능 |
| FOMAML + 자체 PPO | 미검증 | 그래프 재구성 비용으로 실행 못 함 | 🔶 구조 참조 가능 |
| Step 7 GNN 전환 계획 | 미결 (L-D1b) | Hierarchical 폐기 시 부수 삭제 | 🔶 Step 6 후 판단 |

---

## 13. 미결 사항 (Lock)

| ID | 내용 | 해제 조건 |
|---|---|---|
| L-D1b | Step 7 아키텍처 (MLP 순차 vs GNN joint) | Step 5~6 완료 후 |
| L-D1c | 양보(yielding) 메커니즘 | L-D1b 종속 |
| V4 | PBS Φ 후보 비교 | ✅ 완료 — 잠정 맨해튼, V4-3 수렴 G3에서 확정 |
| G3 | Colab 실규모 학습 수렴 확인 | Step 1 착수 전 |
| 서포트 실중량 | Step 6용 현장 데이터 | Step 5 완료 후 |

---

## 14. 다음 실행 순서

```
1. ✅  V4  PBS Φ 후보 비교 완료 (잠정 맨해튼)
2. G3      Colab에서 Step 1 소규모 학습 수렴 확인
           + V4-3 수렴 속도 실측 (맨해튼 vs 복합 비교)
3.         Step 1 정식 학습
4.         Step 1→2 전이 실측 (E2 결과: 90% 보존 예상)
5.         Step 2~6 순차 진행
6. L-D1b  Step 7 아키텍처 결정
```

---

## 15. 검증 결과 요약

| ID | 내용 | 결론 |
|---|---|---|
| **V1** | 벤딩 반경을 action mask로 처리 | ✅ 가능. lookahead k 필수. 경로 팽창 없음 |
| **V2** | JIS 물리 상수 전 구경 | ✅ B/L [0.08,1.05]. Sch40 확정 |
| **V3** | 3D 27방향 계단구조 | ✅ 역방향 자동계산, √2/√3 보정, JIS 범위 유효 |
| **V4** | PBS Φ 후보 비교 | ✅ 잠정 맨해튼 (V4-3 수렴 G3에서 확정) |
| **G1** | J 기반 A* + h(n) 학습 | ✅ 구조 성립. J 정규화 필수 |
| **G2** | sb3-contrib 통합 | ✅ 4개 항목 모두 통과. 환경 버그 2개 수정 |
| **G3** | Colab 실규모 수렴 | 🔜 미실행 |
| **E1** | B/L 계단구조 (2D) | ✅ 해 2종, 전환 [0,1.5]m |
| **E2** | Step 전이 행동 보존율 | ✅ B/L=8.0에서도 90% 보존 |
| **E3** | PBS admissibility | ✅ 조건 준수 시 20/20 최적정책 불변 |

---

*이 문서는 검증이 추가될 때마다 갱신한다.*
*추측을 확정으로 적지 않는다.*
