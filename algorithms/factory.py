import argparse
import importlib
import inspect
from types import SimpleNamespace

SUPPORTED_ALGORITHMS = ("das", "mv", "esbmv", "gcfmv", "cmsaw", "fdmas")
NON_ALGORITHM_FIELDS = {"help", "select_angles", "dr", "tgc", "tgc_alpha", "h5_path", "h5_sample_idx", "output_dir", "save_gt"}
SHARED_ALGORITHM_FIELDS = {"f_number", "dynamic_aperture", "window", "interp", "row_block", "force_gpu_eigh"}


def import_algorithm_module(alg):
    if alg not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"不支持算法 {alg}; 可选 {SUPPORTED_ALGORITHMS}")
    return importlib.import_module(f"algorithms.{alg}")


def algorithm_default_parameters(alg):
    module = import_algorithm_module(alg)
    return {action.dest: module.parser.get_default(action.dest) for action in module.parser._actions
            if action.dest not in NON_ALGORITHM_FIELDS | SHARED_ALGORITHM_FIELDS}


def find_beamformer_class(module, alg):
    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name.endswith("BeamformerIQ"):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"算法 {alg} 没有找到 *BeamformerIQ 类；打包 teacher 需要类的 __call__ 返回 I_out, Q_out。"
        )
    names = [cls.__name__ for cls in candidates]
    raise ValueError(
        f"算法 {alg} 找到多个 BeamformerIQ 类 {names}，请保留一个用于打包的 *BeamformerIQ 类。"
    )


def validate_pack_time_algorithms(used_algs, algorithms, algorithm_params=None):
    known = set(algorithms)
    unsupported = sorted(set(used_algs) - known)
    if unsupported:
        raise ValueError(
            f"outputs 使用了未知算法: {unsupported}; 请先在 algorithms 中添加算法名。"
        )

    for alg, params in (algorithm_params or {}).items():
        if "temporal_win" in params:
            value = params["temporal_win"]
            if type(value) is not int or value < 1 or value % 2 == 0:
                raise ValueError(f"{alg}.temporal_win 必须为正奇数")

    for alg in sorted(set(used_algs)):
        module = import_algorithm_module(alg)
        find_beamformer_class(module, alg)
        actions = module.parser._actions
        supported = {action.dest for action in actions} - NON_ALGORITHM_FIELDS
        params = (algorithm_params or {}).get(alg, {})
        unsupported = set(params) - supported
        if unsupported:
            raise ValueError(f"algorithm_params.{alg} 包含不支持的字段: {sorted(unsupported)}")
        for action in actions:
            if action.dest not in params or action.dest == "temporal_win":
                continue
            value = params[action.dest]
            if action.nargs == 0 and isinstance(action.default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"algorithm_params.{alg}.{action.dest} 必须是布尔值")
            elif action.type is not None:
                try:
                    value = action.type(str(value))
                except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
                    raise ValueError(f"algorithm_params.{alg}.{action.dest}: {exc}") from exc
            if action.choices is not None and value not in action.choices:
                raise ValueError(f"algorithm_params.{alg}.{action.dest} 不在允许选项中")
            params[action.dest] = value


def build_init_kwargs(beamformer_cls, alg_params):
    signature = inspect.signature(beamformer_cls.__init__)
    accepted = set(signature.parameters) - {"self"}
    init_kwargs = {}

    for key, value in (alg_params or {}).items():
        if key in accepted:
            init_kwargs[key] = value

    return init_kwargs


def build_external_beamformer(
    alg,
    z_grid,
    x_grid,
    num_channels,
    pitch,
    c,
    fs,
    t0_scalar,
    input_angles,
    fc,
    params,
    alg_params,
):
    module = import_algorithm_module(alg)
    defaults = {action.dest: module.parser.get_default(action.dest) for action in module.parser._actions if action.dest != "help"}
    defaults.update({key: value for key, value in params.items() if key in defaults})
    defaults.update(alg_params)
    options = SimpleNamespace(**defaults)

    beamformer_cls = find_beamformer_class(module, alg)
    init_kwargs = build_init_kwargs(beamformer_cls, defaults)
    signature = inspect.signature(beamformer_cls.__init__)
    constructor_args = {
        "z_grid": z_grid,
        "x_grid": x_grid,
        "n_elem": num_channels,
        "pitch": pitch,
        "c": c,
        "fc": fc,
        "fs": fs,
        "_t0_all": t0_scalar,
        "t0_all": t0_scalar,
        "angles_rad": input_angles,
    }
    constructor_args = {
        key: value for key, value in constructor_args.items() if key in signature.parameters
    }
    constructor_args.update(init_kwargs)
    constructor_args["options"] = options
    return beamformer_cls(**constructor_args)
