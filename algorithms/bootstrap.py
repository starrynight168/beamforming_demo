from __future__ import annotations

import sys
from pathlib import Path


def initialize_project_path(script_file: str | Path) -> Path:
    project_root = Path(script_file).resolve().parents[1]
    root_entry = str(project_root)
    if root_entry not in sys.path:
        sys.path.insert(0, root_entry)
    return project_root
