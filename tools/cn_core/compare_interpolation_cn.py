import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.shared.tool_wizard import run_registered_tool


def main():
    run_registered_tool("compare_interpolation.py")


if __name__ == "__main__":
    main()
