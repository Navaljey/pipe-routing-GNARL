# D-o. 비공개 서포트 표 — 접근 절차와 하드 스톱

**작성:** 2026-10-19 · **대상:** Colab 에서 라벨 생성·평가를 돌릴 때
**근거:** 2026-10-18 skeleton 사고 (환경 변수 누락 → 공개 임시 표로 채점, PR #26 정정)

> ⚠️ **이 문서에는 사내 표의 수치·문서번호·사진을 넣지 않는다 (D78).**
> 접근 방법과 검증 절차만 다룬다.

---

## 0. 왜 이 절차가 필요한가

skeleton 이 당한 사고의 구조가 중요하다.

```
컨테이너 재시작 → 환경 변수 PIPE_SUPPORT_STANDARD 소실
   → 공개 임시 표(L50 한 종류 근사)로 채점
   → 조용히 통과. 아무 오류도 없었다
   → 나중에 발견: dense50_000 J_total 86,440 → 67,009 kg
                 밀집 50 통과 189 → 185, 서포트 비중 5.5% → 9.2%
```

**시작할 때 한 번 확인하는 것으로는 못 잡는다.** 사고 원인이 *중간에* 환경이
바뀐 것이기 때문이다. Colab 은 세션이 끊기고 재연결되므로 **같은 일이 더 쉽게 난다.**

→ 그래서 확인을 **두 군데**에 둔다.

| 시점 | 무엇을 확인 | 못 잡는 것 |
|---|---|---|
| 시작 전 | 환경 변수 · 파일 존재 · sha256 | 실행 중 환경 변화 |
| **채점 결과마다** | 출력의 `source == "private"` · `sha256` 일치 | — |

두 번째가 핵심이다. 첫 번째만 두면 skeleton 사고가 그대로 재현된다.

---

## 1. 접근 — Colab 에서 비공개 레포 가져오기

### 1.1 PAT 를 Colab Secrets 에 저장 (1회)

GitHub 에서 **fine-grained PAT** 발급:

- Repository access → `Navaljey/pipe-support-standard` 만 선택
- Permissions → **Contents: Read-only** 만. 그 외 전부 No access
- 만료일을 짧게 (90일 등). 길게 둘 이유가 없다

Colab 왼쪽 사이드바 **열쇠 아이콘(🔑 보안 비밀)** →

1. **새 보안 비밀 추가**
2. 이름: `GITHUB_PAT`
3. 값: 발급받은 토큰 붙여넣기
4. 해당 노트북의 **"노트북 액세스" 토글 ON**

> 🚫 **PAT 를 채팅·노트북 셀·커밋에 붙여넣지 않는다.** Colab Secrets 는 노트북
> 소스에 남지 않고 Drive 에도 저장되지 않는다. 이것이 유일하게 안전한 보관처다.

### 1.2 tarball 로 받는다 — `git clone` 보다 안전하다

`git clone https://<PAT>@github.com/...` 는 **PAT 를 `.git/config` 에 적어 넣는다.**
이 프로젝트에서 이미 한 번 겪은 문제다 (clone 후 `git remote set-url` 로 지워야 함).

읽기 전용 데이터 의존성이므로 **`.git` 을 아예 만들지 않는 쪽**이 낫다.

```python
# 셀 — 비공개 서포트 표 가져오기 (멱등: 이미 있으면 재다운로드 생략)
import os, pathlib, hashlib, tempfile, subprocess, urllib.request
from google.colab import userdata

OWNER_REPO = 'Navaljey/pipe-support-standard'
REF        = 'main'
DEST       = pathlib.Path('/content/pipe-support-standard')
STD_NAME   = 'support_standard.json'   # ← 실제 파일명을 skeleton 과 맞출 것 (§4)

def fetch_private_standard() -> pathlib.Path:
    std = DEST / STD_NAME
    if std.is_file():
        return std                      # 재연결 후 /content 가 남아 있는 경우
    pat = userdata.get('GITHUB_PAT')    # 없으면 SecretNotFoundError 로 즉시 실패
    req = urllib.request.Request(
        f'https://api.github.com/repos/{OWNER_REPO}/tarball/{REF}',
        headers={'Authorization': f'Bearer {pat}',
                 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'})
    blob = urllib.request.urlopen(req).read()
    del pat                             # 메모리에 오래 두지 않는다
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as f:
        f.write(blob); tmp = f.name
    DEST.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '-xzf', tmp, '-C', str(DEST),
                    '--strip-components=1'], check=True)
    os.remove(tmp)
    if not std.is_file():
        raise RuntimeError(f'표 파일이 레포에 없다: {STD_NAME} — 파일명 확인 (§4)')
    return std

STD_PATH = fetch_private_standard()
os.environ['PIPE_SUPPORT_STANDARD'] = str(STD_PATH)
```

**멱등으로 쓴 이유:** 재연결 후 `Run All` 한 번으로 복구된다. `/content` 가
남아 있으면 다운로드를 건너뛰고, 런타임이 완전히 리셋됐으면 다시 받는다.
어느 쪽이든 환경 변수는 다시 설정된다.

### 1.3 대안 — Drive 에 표만 두기 (PAT 불필요)

PAT 관리가 부담이면 표 파일만 Drive 에 두고 경로를 가리켜도 된다.

```python
from google.colab import drive
drive.mount('/content/drive')
os.environ['PIPE_SUPPORT_STANDARD'] = '/content/drive/MyDrive/<경로>/support_standard.json'
```

**단점:** Drive 사본이 레포와 어긋날 수 있다. → §2 의 sha256 고정이 그걸 잡으므로
실용적으로는 괜찮다. 어긋나면 **중단**되고, 그때 레포에서 다시 받으면 된다.

---

## 2. 하드 스톱 — D-m ②

`step1_train.ipynb` 셀 5 의 §11 체크리스트와 **같은 계열**로 넣는다.
경고 로그가 아니라 **예외를 던지고 멈춘다.**

```python
import hashlib, pathlib, os

# E1 동결 시 skeleton 이 공개하는 값으로 고정한다 (§4). 그때까지는 None 으로 두고
# 실측값을 기록만 한다 — None 이면 존재·접근만 확인하고 일치 검사는 건너뛴다.
EXPECTED_STD_SHA256 = None   # 예: "a1b2...". 해시는 원본을 복원하지 못하므로 공개 가능


def assert_private_standard(expected: str | None = EXPECTED_STD_SHA256) -> str:
    """시작 전 1회. 반환값 = 실측 sha256 (매니페스트에 기록한다)."""
    p = os.environ.get('PIPE_SUPPORT_STANDARD')
    if not p:
        raise RuntimeError('PIPE_SUPPORT_STANDARD 미설정 — 중단. §1 셀을 먼저 실행')
    path = pathlib.Path(p)
    if not path.is_file():
        raise RuntimeError(f'표 파일 없음: {p} — 중단')
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and got != expected:
        raise RuntimeError('표 sha256 불일치 — 중단\n'
                           f'  기대 {expected}\n  실측 {got}')
    return got


def assert_scored_with_private(out: dict, expected: str) -> None:
    """★ 채점 결과마다. 실행 중 환경이 바뀌는 경우를 잡는 유일한 장치."""
    g = (out.get('global') or {}).get('support_standard') or {}
    src, sha = g.get('source'), g.get('sha256')
    if src != 'private':
        raise RuntimeError(f'공개 표로 채점됨 (source={src!r}) — 결과 폐기, 중단')
    if sha != expected:
        raise RuntimeError('다른 표로 채점됨 — 결과 폐기, 중단\n'
                           f'  기대 {expected}\n  실측 {sha}')


def assert_not_throwaway(manifest: dict) -> None:
    """D-m ① — 폐기용 라벨이 학습에 섞이는 것을 막는다."""
    if manifest.get('throwaway'):
        raise RuntimeError('폐기용 라벨이다 — 학습 중단. labels/_throwaway/ 를 확인')


STD_SHA = assert_private_standard()
print(f'[OK] 비공개 표 확인  sha256={STD_SHA[:16]}…')
```

**사용 지점:**

```python
out = score_scenario(...)                     # skeleton 채점 호출
assert_scored_with_private(out, STD_SHA)      # ★ 매번. 빠뜨리면 사고가 재현된다
```

`EXPECTED_STD_SHA256` 을 `None` 으로 둔 동안에도 **`assert_scored_with_private`
는 완전히 동작한다** — `STD_SHA` 가 실측값이고, 실행 중에 표가 바뀌면 불일치로 걸린다.
E1 동결 때 skeleton 값으로 고정하면 세션 간 일관성까지 보장된다.

---

## 3. 라벨 보관 — D78

**라벨에는 비공개 표에서 나온 서포트 무게가 들어간다.** 이 레포는 **공개**다.

| 무엇 | 어디에 |
|---|---|
| 라벨 파일 (`labels/**`) | **Drive 또는 비공개 레포.** 이 레포에 커밋 금지 |
| 매니페스트 (sha256 · skeleton 커밋 · 격자 변형 · 시드 범위) | 라벨과 같은 곳 |
| 고정 sha256 값 · 이 문서 · 체크 코드 | 공개 레포 가능 (해시는 원본을 복원하지 못한다) |

→ `.gitignore` 에 `labels/` 를 추가했다. **그래도 `git add -A` 전에 한 번 보는 습관을
들이는 쪽이 안전하다** — `.gitignore` 는 이미 추적 중인 파일을 막지 못한다.

### 매니페스트 형식

```python
manifest = {
    'throwaway':               False,          # True 면 학습 루프가 중단한다
    'support_standard_sha256': STD_SHA,        # §2 실측값
    'skeleton_commit':         '<git rev-parse HEAD>',
    'grid_variant':            'G0',
    'seed_range':              [30000000, 39999999],   # 학습용. 평가 시드 금지
    'created_at':              '<ISO8601>',
}
```

---

## 4. 시작 전 확인할 것 2건

| | 확인 | 어디서 |
|---|---|---|
| 1 | **표 파일의 실제 경로·파일명** — `STD_NAME` 을 맞춰야 한다. 비공개 레포가 여러 파일로 쪼개져 있으면 `PIPE_SUPPORT_STANDARD` 가 파일인지 디렉터리인지도 확인 | 비공개 레포 또는 skeleton |
| 2 | **고정할 sha256** — skeleton 이 E1 동결 시 공개한다. 그때 `EXPECTED_STD_SHA256` 에 넣는다 | skeleton |

둘 다 skeleton 쪽에 한 줄로 물으면 된다. **1번이 없으면 §1 셀이 돌지 않는다.**

---

## 5. 체크리스트

라벨 생성을 시작하기 전에 전부 ✅ 여야 한다.

```
[ ] fine-grained PAT 발급 (해당 레포 Contents: Read-only, 만료일 짧게)
[ ] Colab Secrets 에 GITHUB_PAT 저장 + 노트북 액세스 토글 ON
[ ] STD_NAME 을 실제 파일명으로 맞춤                        (§4-1)
[ ] §1 셀 실행 → PIPE_SUPPORT_STANDARD 설정됨
[ ] assert_private_standard() 통과, STD_SHA 확보
[ ] assert_scored_with_private() 를 채점 호출마다 넣음        ★ 핵심
[ ] assert_not_throwaway() 를 학습 루프 진입부에 넣음
[ ] labels/ 가 .gitignore 에 있고, git status 에 안 보임
[ ] 매니페스트에 sha256 · skeleton 커밋 · 격자 변형 · 시드 범위 기록
```

> **v2 전례:** 배선 버그 3건(시드 · w1/w2/w3 · 판정 로직)이 전부 "조용히 통과"했고,
> `Using cpu device` 가 로그에 묻혀 7시간을 버렸다. 고친 방법은 전부
> **하드 스톱**이었다 (`REQUIRE_GPU = True`). 여기도 같다 — 경고는 묻힌다.
