import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.shared.interactive import ROOT, output_input, path_input, run


def main():
    h5_path = path_input("拖入点靶 H5 文件", ROOT / "data/point_targets.h5")
    output_dir = output_input("point_target_metrics")
    run("evaluate_point_targets.py", ["--h5_path", h5_path, "--output_dir", output_dir])


if __name__ == "__main__":
    main()
