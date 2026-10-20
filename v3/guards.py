"""D-m 하드 스톱 — 라벨·채점의 출처를 검증한다.

**경고가 아니라 예외를 던진다.** 이 프로젝트에서 비싼 실패는 전부 "조용히 통과"한
것들이었다:

- v2: 배선 버그 3건(시드 · w1/w2/w3 · 판정 로직)이 전부 조용히 통과
- v2: ``Using cpu device`` 가 로그에 묻혀 7시간 손실 → ``REQUIRE_GPU`` 하드 스톱으로 해결
- skeleton 2026-10-18: ``PIPE_SUPPORT_STANDARD`` 미설정 → 공개 임시 표로 조용히 대체.
  dense50_000 J_total 86,440 → 67,009 kg, 밀집 50 통과 189 → 185

근거 문서
    docs/v3_private_standard_setup.md   — D-o 절차, 표 구조, 고정 해시
    CLAUDE.md §13.1                     — D-m / D-o
    CLAUDE.md §11                       — 환경 구현 체크리스트 (같은 계열)

쓰는 곳
    라벨 생성 노트북과 학습 루프가 **같은 모듈을 import 한다.**
    셀에 코드를 복사하면 두 곳이 어긋난다 — v2 배선 버그와 같은 유형이다.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Any, Iterable, Mapping

__all__ = [
    "GuardError",
    "EXPECTED_STD_SHA256",
    "SEED_RANGES",
    "std_sha256",
    "assert_private_standard",
    "assert_scored_with_private",
    "assert_not_throwaway",
    "assert_seed_range",
    "make_manifest",
]


class GuardError(RuntimeError):
    """하드 스톱. 잡아서 넘어가지 말 것 — 넘어가면 장치를 둔 의미가 없다."""


# --------------------------------------------------------------------------
# 비공개 서포트 표 (D-o)
# --------------------------------------------------------------------------

#: skeleton 비공개 표 main = d404cdc (2026-10-20).
#: 해시는 원본을 복원하지 못하므로 공개 레포에 둘 수 있다.
#: None 으로 두면 존재·구조만 확인하고 값 일치는 건너뛴다 (실측값은 여전히 반환된다).
EXPECTED_STD_SHA256 = "ac2ba6a2974b381938212b84867f1d6f34d87e885b2e1da5a0fc302cb19cffb3"

#: 로더가 "비공개 표인가" 를 판정하는 파일. 없으면 공개 임시 표로 **조용히** 대체된다.
_PROBE = pathlib.Path("tables") / "type_A.json"


def std_sha256(root: str | os.PathLike[str]) -> str:
    """skeleton 규약: ``tables/*.json`` 을 **파일명 정렬 순으로 바이트 이어 붙여** SHA-256.

    개별 파일 해시가 아니다. skeleton 출력의
    ``global.support_standard.sha256`` 과 같은 방식이어야 비교가 성립한다.
    """
    tables = pathlib.Path(root) / "tables"
    files = sorted(tables.glob("*.json"))
    if not files:
        raise GuardError(f"tables/*.json 이 없다: {tables} — 비공개 표가 아니다")
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()


def assert_private_standard(expected: str | None = EXPECTED_STD_SHA256) -> str:
    """시작 전 1회. 반환값 = 실측 sha256 (매니페스트에 기록한다).

    ``PIPE_SUPPORT_STANDARD`` 는 **레포 클론의 루트 디렉터리**다. 파일이 아니다.
    """
    root = os.environ.get("PIPE_SUPPORT_STANDARD")
    if not root:
        raise GuardError(
            "PIPE_SUPPORT_STANDARD 미설정 — 중단. "
            "로더가 공개 임시 표로 조용히 대체한다 (2026-10-18 사고)"
        )
    p = pathlib.Path(root)
    if not p.is_dir():
        raise GuardError(
            f"디렉터리가 아니다: {root} — 중단. "
            "파일이 아니라 레포 클론 루트를 가리켜야 한다"
        )
    if not (p / _PROBE).is_file():
        raise GuardError(
            f"{p / _PROBE} 없음 — 중단. 로더가 공개 임시 표로 떨어진다"
        )
    got = std_sha256(p)
    if expected and got != expected:
        raise GuardError(
            "비공개 표 sha256 불일치 — 중단\n"
            f"  기대 {expected}\n"
            f"  실측 {got}\n"
            "  표가 갱신됐다면 skeleton 과 고정값을 맞출 것"
        )
    return got


def assert_scored_with_private(out: Mapping[str, Any], expected: str) -> None:
    """★ **채점 결과마다** 호출한다.

    실행 중에 환경이 바뀌는 경우를 잡는 **유일한** 장치다. 2026-10-18 사고는
    시작 시점이 아니라 *중간에* 컨테이너가 재시작되며 난 것이고, Colab 은
    세션이 끊기고 재연결되므로 같은 일이 더 쉽게 난다.
    """
    g = (out.get("global") or {}).get("support_standard") or {}
    src = g.get("source")
    sha = g.get("sha256")
    if src is None:
        raise GuardError(
            "출력에 global.support_standard.source 가 없다 — 중단. "
            "구버전 스키마이거나 채점이 잘못됐다"
        )
    if src != "private":
        raise GuardError(f"공개 표로 채점됨 (source={src!r}) — 결과 폐기, 중단")
    if sha != expected:
        raise GuardError(
            "다른 표로 채점됨 — 결과 폐기, 중단\n"
            f"  기대 {expected}\n  실측 {sha}"
        )


# --------------------------------------------------------------------------
# 라벨 관리 (D-m ①)
# --------------------------------------------------------------------------


def assert_not_throwaway(manifest: Mapping[str, Any]) -> None:
    """폐기용 라벨이 학습에 섞이는 것을 막는다.

    skeleton 이 "폐기용 라벨 소량 생성 가능 (폐기용 표시 필수)" 를 허용했다.
    **표시만으로는 섞인다** — 그래서 학습 루프가 예외로 멈춘다.
    """
    if manifest.get("throwaway"):
        raise GuardError(
            "폐기용 라벨이다 — 학습 중단. labels/_throwaway/ 를 확인할 것"
        )


#: 계약 §(d). 평가 시드로 학습하면 평가가 오염되고, 그 사실이 조용히 묻힌다.
SEED_RANGES: dict[str, tuple[int, int]] = {
    "eval": (20_260_000, 20_269_999),
    "train": (30_000_000, 39_999_999),
    "valid": (40_000_000, 49_999_999),
}


def assert_seed_range(seeds: int | Iterable[int], purpose: str = "train") -> None:
    """시드가 그 용도의 범위 안인지 확인한다 (계약 §d).

    평가 범위를 학습에 쓰면 **평가가 오염되고 아무도 모른다.** 그 경우를
    따로 짚어 준다.
    """
    if purpose not in SEED_RANGES:
        raise GuardError(f"알 수 없는 용도: {purpose!r} — {sorted(SEED_RANGES)}")
    lo, hi = SEED_RANGES[purpose]
    values = [seeds] if isinstance(seeds, int) else list(seeds)
    for s in values:
        if lo <= s <= hi:
            continue
        where = next(
            (name for name, (a, b) in SEED_RANGES.items() if a <= s <= b), None
        )
        extra = ""
        if where == "eval":
            extra = "  ← 평가 시드다. 학습에 쓰면 평가가 오염된다"
        elif where:
            extra = f"  ← {where} 범위다"
        raise GuardError(
            f"시드 {s} 가 {purpose} 범위 [{lo}, {hi}] 밖 — 중단{extra}"
        )


def make_manifest(
    *,
    std_sha256_value: str,
    skeleton_commit: str,
    grid_variant: str = "G0",
    purpose: str = "train",
    throwaway: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """라벨 세트 매니페스트. 라벨과 **같은 곳**에 둔다 (공개 레포 금지, D78)."""
    from datetime import datetime, timezone

    m: dict[str, Any] = {
        "throwaway": throwaway,
        "support_standard_sha256": std_sha256_value,
        "skeleton_commit": skeleton_commit,
        "grid_variant": grid_variant,
        "purpose": purpose,
        "seed_range": list(SEED_RANGES[purpose]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    m.update(extra)
    return m
