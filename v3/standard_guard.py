"""서포트 선정표 출처 게이트 — **서포트 비용을 읽는 모든 스크립트가 이것을 먼저 부른다.**

왜 이 파일이 따로 있는가
------------------------
2026-10-22 에 **V3 가 2026-10-18 사고와 같은 조건에 걸렸다.** `|FOCAL|` 측정과
h 느슨함 측정이 `layered.py` 의 D57 추정 서포트를 켜고 돌았는데(`support_cost=True`),
`layered.py` 는 `support_std.load_standard()` 를 부르고 그 로더는
**`PIPE_SUPPORT_STANDARD` 가 없으면 공개 임시 표로 조용히 대체**한다 (D78).
그래서 두 측정의 `C*`·서포트 비중·`f` 가 전부 공개 임시 표 값이었다.

`v3/guards.py` 에 하드 스톱(D-m)이 이미 있었지만 **라벨 생성용으로만 생각해
측정 스크립트에 걸지 않았다.** 그것이 실수였다 — 서포트 비용을 읽는 경로는
라벨이든 측정이든 **전부** 같은 게이트를 지나야 한다.

규약
----
* 기본은 **멈춘다.** 비공개 표가 없으면 `GuardError`.
* 진단·구조 판단용으로 공개 임시 표를 쓰려면 **`--allow-public-standard` 를 명시**해야
  하고(skeleton 이 자기 도구에 둔 것과 같은 이름), 그 사실이 **출력 JSON 과 콘솔에
  남는다.** 조용히 지나가는 경로를 만들지 않는다.
* 반환한 딕셔너리를 결과 JSON 의 `support_standard` 에 **그대로 넣는다** — 나중에
  "이 수치가 어느 표로 나왔나" 를 파일만 보고 알 수 있어야 한다.
"""

from __future__ import annotations

import argparse
import os
import sys

from v3.guards import GuardError, assert_private_standard


def add_argument(ap: argparse.ArgumentParser) -> None:
    """측정·생성 스크립트의 파서에 `--allow-public-standard` 를 단다."""
    ap.add_argument("--allow-public-standard", action="store_true",
                    help="비공개 서포트 표 없이 공개 임시 표로 진행한다 (진단·구조 판단용). "
                         "이 결과는 측정·비교·라벨에 쓸 수 없고, 출력에 그대로 기록된다")


def resolve(allow_public: bool, *, what: str = "이 실행") -> dict:
    """표 출처를 확정해 기록용 딕셔너리로 돌려준다. 비공개가 없고 허용도 없으면 멈춘다."""
    root = os.environ.get("PIPE_SUPPORT_STANDARD", "")
    try:
        sha = assert_private_standard()
        return {"source": "private", "sha256": sha, "root_set": bool(root),
                "allow_public_standard": False}
    except GuardError as e:
        if not allow_public:
            raise GuardError(
                f"{what}: 비공개 서포트 표가 없다 — {e}\n"
                "  서포트 비용(D57 추정)을 읽는 경로이므로 공개 임시 표로 돌면 수치가 전부\n"
                "  달라진다 (2026-10-18 사고: 서포트 비중 5.5% → 9.2%).\n"
                "  PIPE_SUPPORT_STANDARD 를 비공개 표 루트로 두거나,\n"
                "  진단 목적이면 --allow-public-standard 를 명시할 것."
            ) from None
        print(f"⚠️  {what}: 공개 임시 표로 진행한다 (--allow-public-standard). "
              f"측정·비교·라벨에 쓸 수 없다.\n    사유: {e}", file=sys.stderr)
        return {"source": "public_temporary", "sha256": None, "root_set": bool(root),
                "allow_public_standard": True,
                "warning": "공개 임시 표 — 서포트 비용이 비공개 표와 다르다. "
                           "이 수치는 구조 판단용이며 측정·비교·라벨에 쓸 수 없다 (D78·D-m)"}


def banner(std: dict) -> str:
    """콘솔·리포트 한 줄 표기."""
    if std["source"] == "private":
        return f"서포트 표: 비공개 (sha256 {std['sha256'][:12]}…) ✅"
    return "서포트 표: ⚠️ 공개 임시 표 — 측정·비교·라벨 불가"
