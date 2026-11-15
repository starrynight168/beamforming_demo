from importlib import import_module

from .interactive import ROOT, ask, choice, number, output_input, path_input, run, sample_input
from .registry import TOOLS
from .run_output import emit


def ask_field(field, state):
    key, label, kind, default = field
    if kind == "file":
        return key, path_input(f"拖入{label}", default, "file")
    if kind == "text":
        return key, ask(label, default)
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
    emit(None, "prepare", f"{spec['name']}；输入 q 返回总菜单。")
    h5_path = path_input("拖入 H5 文件")
    sample = sample_input(h5_path)
    state = {"h5_path": h5_path, "sample_idx": sample}
    args = [f"--{spec.get('h5_argument', 'h5_path')}", h5_path,
            f"--{spec.get('sample_argument', 'sample_idx')}", sample]
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
    args += [f"--{spec.get('output_argument', 'output_dir')}", output]
    run(spec["script"], args)
    result = output if spec.get("direct_output") else output / f"{h5_path.stem}_sample_{sample}"
    emit(result, "done", f"结果目录：{result}")


def run_registered_tool(script):
    spec = next(spec for spec in TOOLS if spec["script"] == script)
    if spec.get("mode") == "native":
        import_module(spec["entry"]).main()
        return
    run_tool(spec)



