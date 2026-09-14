"""
parent_study_ex29
------------------
Pristine archive of Kyle Lesinger & Di Tian's (2025) peer-reviewed parent codebase
from Nature Communications (DOI: 10.1038/s41467-025-62761-3).
"""
import sys
from pathlib import Path

_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
