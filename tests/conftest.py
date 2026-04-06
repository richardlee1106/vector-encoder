# -*- coding: utf-8 -*-

import sys
from pathlib import Path

CURRENT = Path(__file__).resolve()

for candidate in (CURRENT.parents[1], CURRENT.parents[2]):
    if (candidate / 'v26').exists() and (candidate / 'v26_GLM').exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
