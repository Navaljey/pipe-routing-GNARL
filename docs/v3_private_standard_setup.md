# D-o. 비공개 서포트 표 — 접근 절차와 하드 스톱

**작성:** 2026-10-19 · **정정:** 2026-10-20 (skeleton 회신 반영)
**대상:** Colab 에서 라벨 생성·평가를 돌릴 때

> ⚠️ **이 문서에는 사내 표의 수치·문서번호·사진을 넣지 않는다 (D78).**
> 경로·포맷·해시와 검증 절차만 다룬다.

---

## 0. 표가 실제로 어떻게 생겼나

비공개 레포 `Navaljey/pipe-support-standard` (main, 2026-10-20 기준):

```
tables/           ← 채점에 쓰는 것은 여기뿐
  type_A.json     A 캔틸레버 (벽)
  type_B.json     B 브래킷 + 경사재 (벽)
  type_C.json     C 문형 1배관 (바닥·천장)
  type_D1.json    D 형식 L1 = 가로대 폭
  type_D2.json    D 형식 L2 = 다리 높이
  angle_weights.json
figures/          확인 그림 check_A~D.png
work/             사진 판독 원값      ← 채점에 안 씀
tools/            디지털화 도구        ← 채점에 안 씀
source/           원본 사진           ← 채점에 안 씀
README.md
```

**중요한 두 가지 — 2026-10-19 초안이 둘 다 틀렸다.**

| 항목 | 초안 (틀림) | 실제 |
|---|---|---|
| `PIPE_SUPPORT_STANDARD` | 표 **파일** 경로 | **레포 클론의 루트 디렉터리** |
| sha256 | 파일 하나의 해시 | `tables/*.json` 을 **파일명 정렬 순으로 바이트 이어 붙인** 것의 SHA-256 |

**skeleton 로더의 판정 조건:** `<root>/tables/type_A.json` 이 있으면 비공개 표,
없으면 **공개 임시 표로 조용히 떨어진다.** 그래서 하드 스톱이 이 파일의 존재를 직접 본다.
(skeleton 은 이를 오류로 바꾸는 중 — `--allow-public-standard` 명시 옵션일 때만 공개 표 허용)

**고정 해시 (2026-10-20, main = d404cdc):**

```
ac2ba6a2974b381938212b84867f1d6f34d87e885b2e1da5a0fc302cb19cffb3
```

> 해시는 원본을 복원하지 못하므로 공개 레포에 적어도 된다. skeleton 도 E1 동결 시
> 같은 값을 동결한다.

### 파싱은 직접 하지 않는다 (skeleton 권고, 수용)

> V3 는 파싱을 새로 구현하지 말고 skeleton 의 `pipe_routing.support_std.load_standard()`
> 를 그대로 쓸 것 — **심판과 같은 해석 보장**

묶임 술어를 복제하지 않기로 한 것(D-j)과 **같은 이유**다. 심판의 해석을 두 번 구현하면
어긋나고, 어긋나면 **조용히** 틀린다. 스키마를 아래에 적어두는 것은 참조용이지
직접 파싱하라는 뜻이 아니다.

```
type_X.json : {type, status, grid_round_mm, rule, reinforced_note, uncertain_note, rows}
  rows["15A"…"500A"] = {segments:[{L_from, L_to, angle, raw, uncertain}],
                        L_max, L_max_uncertain, reinforced_from, reinforced_uncertain}
  채점에 쓰는 값 = segments 의 L_from·L_to·angle, reinforced_from[0]
angle_weights.json : {status, note, angles: {앵글명: {kg_m, src}}}
```

---

## 1. 왜 이 절차가 필요한가

2026-10-18 skeleton 사고의 구조 (PR #26 정정):

```
컨테이너 재시작 → PIPE_SUPPORT_STANDARD 가 설정되지 않음
   → 로더가 공개 임시 표로 조용히 대체
   → 아무 오류도 없었다
   → 나중에 발견: dense50_000 J_total 86,440 → 67,009 kg
                 밀집 50 통과 189 → 185, 서포트 비중 5.5% → 9.2%
```

> **원인 정정 (2026-10-20):** 표가 사라진 것이 아니다. 표는 레포에 있었고,
> **환경 변수만** 설정되지 않았다. 로더가 조용히 공개 표로 대체하는 것이 핵심이다.

**시작할 때 한 번 확인하는 것으로는 못 잡는다.** 사고 원인이 *중간에* 환경이 바뀐
것이기 때문이다. Colab 은 세션이 끊기고 재연결되므로 **같은 일이 더 쉽게 난다.**

| 시점 | 무엇을 확인 | 못 잡는 것 |
|---|---|---|
| 시작 전 | 디렉터리 · `tables/type_A.json` · 이어붙인 sha256 | 실행 중 환경 변화 |
| **채점 결과마다** | 출력의 `source == "private"` · `sha256` 일치 | — |

두 번째가 핵심이다. 첫 번째만 두면 사고가 그대로 재현된다.

---

## 2. 접근 — 세 가지 경로

### 2.1 ZIP 업로드 (권장 — PAT·Drive 불필요)

회사 컴퓨터에서 Drive 업로드가 막혀 있고 PAT 가 말을 안 들을 때 **오늘 바로 되는 길**이다.

1. 비공개 레포 `main` → **Code → Download ZIP**
2. 아래 셀 실행 → 그 ZIP 선택

```python
# 셀 C — 비공개 표 ZIP 업로드
import os, zipfile, pathlib
from google.colab import files

up = files.upload()                    # pipe-support-standard-main.zip
zipfile.ZipFile(next(iter(up))).extractall('/content/std_zip')

# ZIP 은 <레포명>-main/ 한 겹을 더 만든다 — tables/type_A.json 을 가진 곳을 찾는다
root = next(p.parent.parent
            for p in pathlib.Path('/content/std_zip').rglob('tables/type_A.json'))
os.environ['PIPE_SUPPORT_STANDARD'] = str(root)
print('루트:', root)
```

- 세션이 끊기면 다시 올린다. 클릭 두 번
- ZIP 에 원본 사진도 들어온다. 회사 정책상 걸리면 `tables/` 의 JSON 6개만
  따로 받아 올려도 된다 — 위 `rglob` 가 알아서 찾는다
  (단 그 경우 `tables/` 구조를 만들어 줘야 한다)

### 2.2 PAT + tarball (자동화하려면)

`git clone https://<PAT>@github.com/...` 는 **PAT 를 `.git/config` 에 적어 넣는다.**
읽기 전용 의존성이므로 `.git` 을 아예 만들지 않는 tarball 이 낫다.

**PAT 발급** — Settings → Developer settings → Personal access tokens →
Fine-grained tokens → Generate new token

| 항목 | 값 |
|---|---|
| Expiration | 90 days (짧게) |
| Resource owner | **본인 계정.** 조직을 고르면 승인 대기로 401 이 계속 난다 |
| Repository access | Only select → `pipe-support-standard` |
| Permissions | **Contents: Read-only** 만 |

생성 **직후 화면에서만** 값이 보인다. 목록으로 돌아가면 이름만 보인다.

**Colab Secrets** (사이드바 🔑) → 이름 `GITHUB_PAT` → 값 붙여넣기 →
**노트북 액세스 토글 ON** → **런타임 연결 해제 및 삭제** 후 재실행.

```python
import os, io, pathlib, tempfile, subprocess, urllib.request
from google.colab import userdata

DEST = pathlib.Path('/content/pipe-support-standard')

def fetch_private_standard() -> pathlib.Path:
    if (DEST / 'tables' / 'type_A.json').is_file():
        return DEST                                   # 재연결 후 남아 있으면 생략
    pat = userdata.get('GITHUB_PAT')
    req = urllib.request.Request(
        'https://api.github.com/repos/Navaljey/pipe-support-standard/tarball/main',
        headers={'Authorization': f'Bearer {pat}',
                 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'})
    blob = urllib.request.urlopen(req).read()
    del pat
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as f:
        f.write(blob); tmp = f.name
    DEST.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar','-xzf',tmp,'-C',str(DEST),'--strip-components=1'], check=True)
    os.remove(tmp)
    if not (DEST / 'tables' / 'type_A.json').is_file():
        raise RuntimeError('tables/type_A.json 없음 — 공개 표로 떨어진다. 중단')
    return DEST

os.environ['PIPE_SUPPORT_STANDARD'] = str(fetch_private_standard())
```

**401 이 날 때** — `/user` 로 찔러 토큰 자체가 거부되는지 가른다 (401 = 자격증명 불량,
404 = 토큰은 유효한데 접근 권한 없음). 토큰 목록의 **"Never used"** 표시가 결정적이다 —
새 토큰이 "Never used" 면 Colab 이 **다른(옛) 값**을 보내고 있다는 뜻이다.

### 2.3 Drive — **채택 (2026-10-20).** 업로드가 되는 컴퓨터에서 1회

회사 컴퓨터는 정책상 **파일 업로드 자체가 막혀** 있어 §2.1 ZIP 업로드도 쓸 수 없다.
→ **업로드가 되는 컴퓨터(집 노트북)에서 한 번 Drive 에 올려두고, 그 뒤로는
어디서든 Colab 이 마운트해 읽는다.** Colab ↔ Drive 는 구글 서버끼리의 일이라
회사 PC 를 거치지 않는다.

#### ① 받기 (업로드 되는 컴퓨터)

1. `github.com/Navaljey/pipe-support-standard`, 브랜치 **`main`** 확인
2. **`<> Code` → Download ZIP** → 압축 풀기
3. 쓰는 것은 **`tables/` 뿐**

대안: `tables/` 의 파일마다 **Download raw file** 버튼으로 6개를 받는다 —
`type_A · type_B · type_C · type_D1 · type_D2 · angle_weights` (`.json`)

#### ② Drive 에 올리기

`drive.google.com` → 내 드라이브 → 새 폴더 `pipe-support-standard` →
그 안에서 **폴더 업로드**로 `tables` 폴더를 넣는다.

```
내 드라이브/<원하는 위치>/pipe-support-standard/tables/
    angle_weights.json
    type_A.json  type_B.json  type_C.json  type_D1.json  type_D2.json
```

**상위 폴더는 자유다.** 단 Colab 경로에 그대로 반영해야 한다 — 예를 들어
`내 드라이브 > Papers > pipe-support-standard` 에 두었으면 `ROOT` 는
`/content/drive/MyDrive/Papers/pipe-support-standard` 다. **`tables/` 이름만
고정**이다 (로더가 `<root>/tables/type_A.json` 으로 판정한다).

> ⚠️ **`source/`(원본 사진)는 올리지 않는다.** 채점에 안 쓰이고 D78 상 비공개
> 레포에만 있어야 한다. `figures/`·`work/`·`tools/` 도 불필요하다.

#### ③ Colab 에서 확인

```python
from google.colab import drive
drive.mount('/content/drive')          # 권한 팝업 → 계정 선택 → 허용

import os, pathlib, hashlib
ROOT = '/content/drive/MyDrive/Papers/pipe-support-standard'   # ← 실제 위치로
os.environ['PIPE_SUPPORT_STANDARD'] = ROOT

tables = pathlib.Path(ROOT) / 'tables'
print('tables 존재:', tables.is_dir())

fs = sorted(tables.glob('*.json'))
print('파일 개수:', len(fs), '(6이어야 함)')
for f in fs:
    print(f'   {f.name:24s} {f.stat().st_size:>7,} bytes')

h = hashlib.sha256()
for f in fs:
    h.update(f.read_bytes())
print('해시 :', h.hexdigest())
print('일치 :', h.hexdigest() == EXPECTED_STD_SHA256)
```

경로를 모르면 찾는다 (Drive 가 크면 1~2분):

```python
for p in pathlib.Path('/content/drive/MyDrive').glob('**/tables/type_A.json'):
    print('ROOT =', p.parent.parent)
```

> 📌 경로는 영문 **`MyDrive`** 다. 화면에 "내 드라이브" 로 보여도
> `/content/drive/내 드라이브/…` 는 작동하지 않는다.

#### ④ 해시가 안 맞을 때 — 함정 셋

| 함정 | 결과 |
|---|---|
| **미리보기 화면에서 복사-붙여넣기** | 줄바꿈·공백·인코딩이 달라져 바이트가 바뀐다. ZIP 이나 Download raw file 로만 받는다 |
| **에디터로 열어서 저장(Ctrl+S)** | Windows 에서 LF → CRLF 로 바뀌어 해시가 달라진다. **보기만 하고 저장하지 않는다** |
| **파일 누락** | 6개여야 한다. `angle_weights.json` 은 `type_*` 패턴이 아니라 특히 빠뜨리기 쉽다 |
| **경로 오타** | `tables 존재: False` 또는 파일 0개. 해시가 `e3b0c442…` 로 시작하면 **빈 입력**이라는 뜻이다 (2026-10-20 실제로 걸렸다 — 상위 폴더 `Papers/` 누락) |

#### ⑤ 이후 — 재업로드 없음

| | 빈도 |
|---|---|
| 파일 업로드 | **1회** |
| `drive.mount()` + 권한 승인 | 세션마다 (클릭 3번) |
| 해시 확인 | 자동 (§3 하드 스톱) |

> 사내 표를 개인 Drive 에 두는 것이 회사 정책상 허용되는지는 소유자 판단이다.

---

## 3. 하드 스톱 — D-m ②

`§11 체크리스트`와 **같은 계열**로 넣는다. 경고 로그가 아니라 **예외를 던지고 멈춘다.**

```python
import os, hashlib, pathlib

EXPECTED_STD_SHA256 = 'ac2ba6a2974b381938212b84867f1d6f34d87e885b2e1da5a0fc302cb19cffb3'


def std_sha256(root) -> str:
    """skeleton 규약: tables/*.json 을 파일명 정렬 순으로 바이트 이어 붙여 SHA-256."""
    fs = sorted((pathlib.Path(root) / 'tables').glob('*.json'))
    if not fs:
        raise RuntimeError(f'tables/*.json 없음: {root}')
    h = hashlib.sha256()
    for f in fs:
        h.update(f.read_bytes())
    return h.hexdigest()


def assert_private_standard(expected: str | None = EXPECTED_STD_SHA256) -> str:
    """시작 전 1회. 반환값 = 실측 sha256 (매니페스트에 기록한다)."""
    root = os.environ.get('PIPE_SUPPORT_STANDARD')
    if not root:
        raise RuntimeError('PIPE_SUPPORT_STANDARD 미설정 — 중단')
    p = pathlib.Path(root)
    if not p.is_dir():
        raise RuntimeError(f'디렉터리가 아니다: {root} — 중단 '
                           '(파일이 아니라 레포 클론 루트를 가리켜야 한다)')
    if not (p / 'tables' / 'type_A.json').is_file():
        raise RuntimeError(f'{p}/tables/type_A.json 없음 — '
                           '로더가 공개 임시 표로 떨어진다. 중단')
    got = std_sha256(p)
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

---

## 4. 라벨 보관 — D78

**라벨에는 비공개 표에서 나온 서포트 무게가 들어간다.** 이 레포는 **공개**다.

| 무엇 | 어디에 |
|---|---|
| 라벨 파일 (`labels/**`) | **Drive 또는 비공개 레포.** 이 레포에 커밋 금지 |
| 매니페스트 | 라벨과 같은 곳 |
| 고정 sha256 · 이 문서 · 체크 코드 | 공개 레포 가능 |

`.gitignore` 에 `labels/` · `support_standard*.json` · `*.label.json` 을 넣었다.
**`.gitignore` 는 이미 추적 중인 파일을 막지 못하므로** 첫 라벨 뒤에 `git status` 를 한 번 본다.

```python
manifest = {
    'throwaway':               False,
    'support_standard_sha256': STD_SHA,
    'skeleton_commit':         '<git rev-parse HEAD>',
    'grid_variant':            'G0',
    'seed_range':              [30000000, 39999999],   # 학습용. 평가 시드 금지
    'created_at':              '<ISO8601>',
}
```

---

## 5. 체크리스트

```
[ ] 비공개 레포 main 에 tables/*.json 6개가 보인다       (2026-10-20 병합 완료)
[ ] §2 셋 중 하나로 PIPE_SUPPORT_STANDARD = 루트 디렉터리 설정
[ ] assert_private_standard() 통과 · 해시가 고정값과 일치
[ ] 파싱은 skeleton 의 load_standard() 를 쓴다 (직접 구현 금지)
[ ] assert_scored_with_private() 를 채점 호출마다 넣음        ★ 핵심
[ ] assert_not_throwaway() 를 학습 루프 진입부에 넣음
[ ] labels/ 가 .gitignore 에 있고 git status 에 안 보인다
[ ] 매니페스트에 sha256 · skeleton 커밋 · 격자 변형 · 시드 범위 기록
```

> **v2 전례:** 배선 버그 3건(시드 · w1/w2/w3 · 판정 로직)이 전부 "조용히 통과"했고,
> `Using cpu device` 가 로그에 묻혀 7시간을 버렸다. 고친 방법은 전부 **하드 스톱**이었다.
> 여기도 같다 — 경고는 묻힌다.
