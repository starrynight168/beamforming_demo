import sys
import runpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    runpy.run_module("tools.code.normalization_demo", run_name="__main__")


if __name__ == "__main__":
    main()
