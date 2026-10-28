import h5py

from .interactive import ROOT, choice, number, output_input, path_input, run, sample_input
from .registry import TOOLS


def ask_field(field, state):
    key, label, kind, default = field
    if kind == "rf_source":
        inferred = ROOT / "data/EPFL" / state["h5_path"].stem
        path = path_input(f"拖入{label}", inferred if inferred.is_dir() else None, "either")
        if path.is_file() and path.suffix.lower() != ".npz":
            raise ValueError("原始 RF 文件必须为 NPZ。")
        return ("rf_path" if path.is_file() else "rf_root"), path
    if kind == "settings":
        path = path_input(f"拖入{label}", default, "dir")
        for name in ("steering_angles.npy", "time_axis.npy"):
            if not (path / name).is_file():
                raise ValueError(f"settings 文件夹缺少 {name}")
        return key, path
    if kind == "reference":
        with h5py.File(state["h5_path"], "r") as f:
            fields = [name for name in ("all_envdb_das_norm", "all_envdb_norm") if name in f]
        if not fields:
            raise ValueError("H5 缺少可用的归一化参考图。")
        return key, choice(label, [(name, name) for name in fields])
    if kind in {"choice", "flag"}:
        return key, choice(label, default)
    rules = {
        "positive": (float, lambda v: v > 0),
        "nonnegative": (float, lambda v: v >= 0),
        "ratio": (float, lambda v: 0 < v <= 1),
        "positive_int": (int, lambda v: v > 0),
        "odd": (int, lambda v: v > 0 and v % 2 == 1),
        "sections": (int, lambda v: v >= 3),
        "sound_max": (float, lambda v: v > state["c_min"]),
    }
    cast, valid = rules[kind]
    return key, number(label, default, cast, valid)


def run_tool(spec):
    print(f"\n{spec['name']}：路径可直接拖入窗口；输入 q 返回总菜单。")
    h5_path = path_input("拖入 H5 文件")
    sample = sample_input(h5_path)
    state = {"h5_path": h5_path, "sample_idx": sample}
    args = ["--h5_path", h5_path, "--sample_idx", sample]
    for field in spec["basic"]:
        key, value = ask_field(field, state)
        state[key] = value
        args += [f"--{key}", value]
    if choice("参数设置", [(False, "使用默认值"), (True, "手动设置")]):
        for field in spec["advanced"]:
            key, value = ask_field(field, state)
            state[key] = value
            if field[2] == "flag":
                if value:
                    args.append(f"--{key}")
            else:
                args += [f"--{key}", value]
    output = output_input(spec["output_name"])
    args += ["--output_dir", output]
    run(spec["script"], args)
    print(f"\n结果目录: {output / f'{h5_path.stem}_sample_{sample}'}")


def main():
    while True:
        print("\n超声工具总入口")
        for index, spec in enumerate(TOOLS, 1):
            print(f"  {index}. {spec['name']}")
        print("  0. 退出")
        try:
            selected = input("请选择工具: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        if selected in {"0", "q", "quit", "exit", "退出"}:
            print("已退出。")
            return
        if not selected.isdigit() or not 1 <= int(selected) <= len(TOOLS):
            print("选择无效，请重新输入。")
            continue
        try:
            run_tool(TOOLS[int(selected) - 1])
        except SystemExit as exc:
            print(exc)
        except (OSError, ValueError) as exc:
            print(f"无法运行：{exc}")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return

