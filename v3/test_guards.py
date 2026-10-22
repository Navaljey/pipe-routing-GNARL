"""v3/guards.py 단위 시험 — **정말 멈추는지**를 확인한다.

경고만 내고 지나가면 장치를 둔 의미가 없으므로, 각 실패 경로가 GuardError 를
던지는 것을 하나씩 박아 둔다. 비공개 표도 PAT 도 필요 없다 — 가짜 데이터로 돈다.

    python3 v3/test_guards.py        (pytest 없이도 실행된다)
    pytest v3/test_guards.py
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from v3.guards import (  # noqa: E402
    GuardError,
    SEED_RANGES,
    assert_not_throwaway,
    assert_private_standard,
    assert_scored_with_private,
    assert_seed_range,
    make_manifest,
    std_sha256,
)

# --------------------------------------------------------------------------
# 도우미
# --------------------------------------------------------------------------

_NAMES = [
    "angle_weights.json",
    "type_A.json",
    "type_B.json",
    "type_C.json",
    "type_D1.json",
    "type_D2.json",
]


def _fake_standard(tmp: pathlib.Path) -> tuple[pathlib.Path, str]:
    """실제 표가 아닌 **가짜** 표. 구조만 같다 (사내 수치 없음)."""
    tables = tmp / "tables"
    tables.mkdir(parents=True)
    for n in _NAMES:
        (tables / n).write_text(json.dumps({"stub": n}), encoding="utf-8")
    h = hashlib.sha256()
    for f in sorted(tables.glob("*.json")):
        h.update(f.read_bytes())
    return tmp, h.hexdigest()


def _raises(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except GuardError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} 가 멈추지 않았다 — 하드 스톱이 아니다")


def _out(source="private", sha="abc", with_key=True) -> dict:
    g: dict = {}
    if with_key:
        g = {"source": source, "sha256": sha}
    return {"global": {"support_standard": g}}


# --------------------------------------------------------------------------
# 1. 비공개 표 (D-o)
# --------------------------------------------------------------------------


def test_정상_경로는_통과하고_해시를_반환한다():
    with tempfile.TemporaryDirectory() as d:
        root, sha = _fake_standard(pathlib.Path(d))
        os.environ["PIPE_SUPPORT_STANDARD"] = str(root)
        assert assert_private_standard(expected=sha) == sha
        assert assert_private_standard(expected=None) == sha  # 고정값 없이도 동작


def test_환경변수_없으면_멈춘다():
    os.environ.pop("PIPE_SUPPORT_STANDARD", None)
    assert "미설정" in _raises(assert_private_standard)


def test_디렉터리가_아니면_멈춘다():
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        os.environ["PIPE_SUPPORT_STANDARD"] = f.name
        assert "디렉터리가 아니다" in _raises(assert_private_standard)


def test_tables_type_A가_없으면_멈춘다():
    """로더가 공개 임시 표로 조용히 떨어지는 바로 그 조건."""
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "tables").mkdir()
        os.environ["PIPE_SUPPORT_STANDARD"] = d
        assert "type_A.json" in _raises(assert_private_standard)


def test_해시가_다르면_멈춘다():
    with tempfile.TemporaryDirectory() as d:
        root, _ = _fake_standard(pathlib.Path(d))
        os.environ["PIPE_SUPPORT_STANDARD"] = str(root)
        assert "불일치" in _raises(assert_private_standard, expected="0" * 64)


def test_해시는_정렬_후_이어붙인_것이다():
    """개별 파일 해시가 아니다 — skeleton 규약과 같아야 비교가 성립한다."""
    with tempfile.TemporaryDirectory() as d:
        root, sha = _fake_standard(pathlib.Path(d))
        manual = hashlib.sha256()
        for n in sorted(_NAMES):  # 파일명 정렬 순
            manual.update((root / "tables" / n).read_bytes())
        assert std_sha256(root) == manual.hexdigest() == sha


def test_tables가_비면_멈춘다():
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "tables").mkdir()
        assert "tables/*.json" in _raises(std_sha256, d)


# --------------------------------------------------------------------------
# 2. 채점 출처 — 실행 중 환경 변화를 잡는 장치
# --------------------------------------------------------------------------


def test_private이고_해시가_같으면_통과():
    assert_scored_with_private(_out("private", "abc"), "abc")


def test_공개표로_채점되면_멈춘다():
    """2026-10-18 사고가 바로 이것. 당시엔 아무 오류도 안 났다."""
    assert "공개 표로 채점됨" in _raises(
        assert_scored_with_private, _out("public", "abc"), "abc"
    )


def test_source_키가_없으면_멈춘다():
    """구버전 스키마를 '통과' 로 읽으면 안 된다."""
    assert "source 가 없다" in _raises(
        assert_scored_with_private, _out(with_key=False), "abc"
    )


def test_global이_통째로_없어도_멈춘다():
    assert "source 가 없다" in _raises(assert_scored_with_private, {}, "abc")


def test_해시가_다른_표로_채점되면_멈춘다():
    assert "다른 표로 채점됨" in _raises(
        assert_scored_with_private, _out("private", "zzz"), "abc"
    )


# --------------------------------------------------------------------------
# 3. 폐기용 라벨 (D-m ①)
# --------------------------------------------------------------------------


def test_폐기용_라벨이면_멈춘다():
    assert "폐기용" in _raises(assert_not_throwaway, {"throwaway": True})


def test_정상_라벨은_통과():
    assert_not_throwaway({"throwaway": False})
    assert_not_throwaway({})  # 키가 없으면 정상으로 본다


# --------------------------------------------------------------------------
# 4. 시드 범위 (계약 §d)
# --------------------------------------------------------------------------


def test_학습_시드는_통과():
    assert_seed_range(30_000_000)
    assert_seed_range(39_999_999)
    assert_seed_range([30_000_001, 35_000_000])


def test_평가_시드를_학습에_쓰면_멈추고_이유를_말한다():
    msg = _raises(assert_seed_range, 20_260_000, "train")
    assert "평가 시드" in msg and "오염" in msg


def test_검증_시드를_학습에_쓰면_멈춘다():
    assert "valid 범위" in _raises(assert_seed_range, 40_000_000, "train")


def test_어느_범위도_아니면_멈춘다():
    assert "밖" in _raises(assert_seed_range, 123, "train")


def test_용도_오타는_멈춘다():
    assert "알 수 없는 용도" in _raises(assert_seed_range, 30_000_000, "trian")


def test_범위들은_겹치지_않는다():
    rs = sorted(SEED_RANGES.values())
    for (a1, b1), (a2, _) in zip(rs, rs[1:]):
        assert b1 < a2, f"시드 범위가 겹친다: {a1}-{b1} / {a2}-"


# --------------------------------------------------------------------------
# 5. 매니페스트
# --------------------------------------------------------------------------


def test_매니페스트에_필수_항목이_들어간다():
    m = make_manifest(std_sha256_value="abc", skeleton_commit="deadbee")
    for k in ("throwaway", "support_standard_sha256", "skeleton_commit",
              "grid_variant", "seed_range", "created_at"):
        assert k in m, k
    assert m["throwaway"] is False
    assert m["seed_range"] == list(SEED_RANGES["train"])
    assert_not_throwaway(m)          # 기본값은 학습에 쓸 수 있어야 한다


def test_폐기용_매니페스트는_학습을_막는다():
    m = make_manifest(std_sha256_value="abc", skeleton_commit="x", throwaway=True)
    assert "폐기용" in _raises(assert_not_throwaway, m)


# --------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                     # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 통과")
    sys.exit(1 if failed else 0)
