"""
Moved: scripts/benchmarks/bench_compare.py

This file has been relocated to keep the repository root clean.
Attempting to run the new script automatically for backward compatibility.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_PATH = os.path.join(HERE, "scripts", "benchmarks", "bench_compare.py")

if os.path.exists(NEW_PATH):
    sys.argv[0] = NEW_PATH  # make argparse-friendly if used later
    runpy.run_path(NEW_PATH, run_name="__main__")
else:
    raise SystemExit(
        "bench_compare.py has moved to scripts/benchmarks/bench_compare.py"
    )
