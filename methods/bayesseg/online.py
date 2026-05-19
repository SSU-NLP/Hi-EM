#!/usr/bin/env python3
"""BayesSeg **online** (prefix-causal, native-K, persistent JVM) — AUXILIARY.

Hi-EM 수정본. 단일 진실원천 = ``scripts/run_bayesseg_prefix.py``
(OnlineSegServer persistent JVM, num-segs-known=false native-K,
검증 완료). 코드 중복 없이 그 러너를 그대로 실행하는 진입점.

  python methods/bayesseg/online.py [--name ... --datasets ... ...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent.parent
          / "scripts" / "run_bayesseg_prefix.py")

if __name__ == "__main__":
    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT), run_name="__main__")
