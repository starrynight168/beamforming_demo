"""MBAN 训练与评估中文向导；评估只生成选择计划并调用统一入口。"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from eval_core.config import (
    DEFAULT_CONFIG as DEFAULT_TRAIN_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    RESULTS_DIR as DEFAULT_RESULTS_DIR,
    ROOT,
    TRAIN_DIR,
    resolve_hardware_config,
    resolve_path,
)
from evaluate_mban import load_eval

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


TRAIN_SCRIPT = TRAIN_DIR / "mban.py"
VALIDATE_SCRIPT = TRAIN_DIR / "evaluate_mban.py"
DEFAULT_EVAL = TRAIN_DIR / "eval.yaml"
SELECTED_EVAL = TRAIN_DIR / "results" / "eval" / "_plans" / "eval_selected.yaml"
ACTIVE_CONFIG_BASE_DIR = DEFAULT_TRAIN_CONFIG.parent

class BackCommandError(Exception):
    """Return to the previous interactive step."""


class ExitCommandError(Exception):
    """Exit the current wizard."""


HELP = {
    "workflow": "训练生成独立 YAML 后调用 mban.py；评估调用 evaluate_mban.py，所有模式都复用 train 目录中的统一实现。",
    "base": "向导以现有训练 YAML 为模板，只修改你选择的字段，不覆盖原始 config.yaml。",
    "dynamic_aperture": "true 时按深度和 F-Number 计算有效接收孔径，并居中打包到固定 network_channels；false 使用全部物理通道。",
    "output_controls": "网络原生输出 K 个控制点，再插值到当前有效孔径；0 表示 Direct-Full，输出完整网络孔径宽度。",
    "projection_controls": "Direct 全维输出后在有效孔径内执行 M→Kp→M 投影；它和原生低维 output_controls 不是同一个模型。",
    "output_weights": "real 输出实权重；complex 输出实部和虚部。nonnegative 只支持 real。",
    "hidden_layers": "隐藏层数量，输出层自动为 fc{hidden_layers+1}。代码支持任意正整数，论文协议常用 1、2、3。",
    "normalization": "none、l1、l2、running_zscore、batch_renorm 五选一；运行统计归一化含均值中心化和方差缩放；归一化作用层由 normalization_layers 指定。",
    "normalization_layers": "执行归一化的隐藏层，例如 [fc2]；不能填写输出层。",
    "weight_transform": "独立权重处理：none 或 WS；不属于隐藏激活归一化。",
    "weight_transform_layers": "WS 的作用层，例如 [fc1, fc2, fc3]；留空时作用于全部 FC 层。",
    "weight_transform_epsilon": "WS 的数值稳定 epsilon。",
    "centering": "none 或 sample_mean；sample_mean 在指定隐藏层的 dual/L1/L2 之前按样本减均值。",
    "centering_layers": "sample_mean 的作用层，例如 [fc1, fc2]；必须是已配置激活函数的隐藏层。",
    "branch_mode": "single 保留单路；dual 将隐藏特征拆成正、负两条 rail，并使用联合归一化。",
    "target_algorithm": "从 H5 的 config_yaml.generation.training_targets.complex_iq_targets 选择复数 IQ 监督目标；当前主线为 mv。",
    "loss.error_function": "mse、l1、charbonnier 统一用于 I/Q 和线性包络误差。",
    "resume": "none 从头训练；weights 只加载模型权重；all 恢复模型、优化器、调度器和训练轮次。",
    "mode": "software 是 FP32 训练；qat 是量化感知训练。ptq 只用于评估，不可作为 mban.py 的训练模式。",
    "evaluation": "评估计划统一选择一个 model_type 和多个评估项；复杂参数继续由 eval.yaml 维护。",
    "checkpoint": "论文和部署主 checkpoint 是 best_val.pth；latest.pth 主要用于断点续训；best_train.pth 不作为主结果。",
    "baselines": "场景评估默认重算 DAS/MV；跳过可用于快速冒烟测试，但正式结果不建议跳过。",
    "hardware": "硬件模式仍由 evaluate_mban.py 统一执行，区分 PTQ/QAT、非理想 stress、mapping、CrossSim 和 PPA ledger。",
    "scheduler": "none 固定学习率；其余调度器只在下方询问各自相关的子参数。",
    "output_domain": "输出域：nonnegative 非负权重，signed 有符号权重。",
    "unity_constraint": "unity 约束：hard 为硬归一化，free 为自由权重。",
    "input_normalization": "none、rms、std 三选一；对复 IQ 做通道归一化，通道求和后自动恢复输出幅度。",
    "loss.terms.d1.domain": "D1 作用域：shape 尺度无关，absolute 使用绝对权重。",
    "loss.terms.d1.weight": "D1 一阶差分正则系数；0 关闭。",
    "loss.terms.d2.domain": "D2 作用域：shape 尺度无关，absolute 使用绝对权重。",
    "loss.terms.d2.weight": "D2 二阶差分正则系数；0 关闭。",
    "loss.terms.lrange.limit": "raw-control 范围阈值。",
    "loss.terms.lrange.weight": "raw-control 范围正则系数；0 关闭。",
    "output_directory": "结构、孔径、控制点数或损失语义变化时使用新的输出目录；latest.pth 仅用于继续训练。",
    "interpolation_bits": "K→M 插值系数的定点精度，只影响硬件/部署边界；可选 2–32 bit。",
    "quantization": "QAT 才启用硬件量化字段；PTQ 是 evaluate_mban.py 的评估模式，不是训练模式。",
    "target": "监督目标必须是 H5 元数据中存在的复数 IQ 算法；IQ 与包络损失不能同时关闭。",
    "data": "每个 H5 独立按 train/val/test 比例划分；固定 split 会校验样本标签和比例，正式实验应复用同一 CSV。",
}


CHOICES = {
    "scheduler": ("none", "cosine", "step", "multistep", "onecycle", "plateau"),
    "resume": ("none", "weights", "all"),
    "error_function": ("mse", "l1", "charbonnier"),
    "centering": ("none", "sample_mean"),
    "normalization": ("none", "l1", "l2", "running_zscore", "batch_renorm"),
    "weight_transform": ("none", "ws"),
    "output_domain": ("nonnegative", "signed"),
    "unity_constraint": ("hard", "free"),
    "output_weights": ("real", "complex"),
    "branch_mode": ("single", "dual"),
    "input_normalization": ("none", "rms", "std"),
    "output_interpolation": ("linear", "nearest", "cubic"),
    "inter_layer": ("none", "analog", "digital"),
    "inter_layer_bits": tuple(str(bits) for bits in range(2, 33)),
    "interpolation_bits": tuple(str(bits) for bits in range(2, 33)),
    "run.mode": ("software", "qat", "ptq"),
    "activation_type": ("relu", "pwl", "tanh"),
    "noise_mode": ("local_relative", "full_scale"),
    "tia_saturation_mode": ("absolute", "full_scale"),
    "activation_threshold_mode": ("absolute", "full_scale"),
    "domain": ("shape", "absolute"),
}


# 与 CHOICES 同键的选项中文解释；缺失时回退为选项值本身。
CHOICE_DESCRIPTIONS = {
    "scheduler": {
        "none": "固定学习率",
        "cosine": "余弦退火（可设重启周期）",
        "step": "按固定间隔衰减",
        "multistep": "在指定轮次衰减",
        "onecycle": "单周期循环学习率",
        "plateau": "指标停滞时衰减",
    },
    "resume": {
        "none": "从头训练（可选 initial_checkpoint 热启动）",
        "weights": "只恢复模型权重",
        "all": "恢复模型、优化器、调度器和轮次",
    },
    "error_function": {
        "mse": "均方误差",
        "l1": "L1 绝对误差",
        "charbonnier": "Charbonnier 平滑误差",
    },
    "normalization": {
        "none": "不归一化",
        "l1": "L1 归一化到总和 1（仅 dual）",
        "l2": "L2 归一化",
        "running_zscore": "运行统计 Z-Score（需设置动量/epsilon）",
        "batch_renorm": "Batch Renormalization（训练批校正，部署可折叠）",
    },
    "weight_transform": {
        "none": "不处理权重",
        "ws": "Weight Standardization（逐输出通道标准化）",
    },
    "centering": {
        "none": "不做隐藏层逐样本中心化",
        "sample_mean": "按样本特征均值中心化（位于 dual/L1/L2 之前）",
    },
    "output_domain": {
        "nonnegative": "非负输出（ReLU）",
        "signed": "有符号输出",
    },
    "unity_constraint": {
        "hard": "硬 unity 约束",
        "free": "自由输出，使用软 unity 损失",
    },
    "output_weights": {
        "real": "实权重（nonnegative 仅支持 real）",
        "complex": "复权重（实部 + 虚部）",
    },
    "branch_mode": {
        "single": "单路隐藏特征",
        "dual": "正负双 rail + 联合归一化",
    },
    "input_normalization": {
        "none": "不做通道归一化",
        "rms": "按 RMS 归一化",
        "std": "按标准差归一化",
    },
    "output_interpolation": {
        "linear": "线性插值",
        "nearest": "最近邻插值",
        "cubic": "四点 Catmull–Rom 三次插值",
    },
    "inter_layer": {
        "analog": "模拟层间通路",
        "digital": "数字层间通路",
    },
    "run.mode": {
        "software": "FP32 软件训练",
        "qat": "量化感知训练",
        "ptq": "仅用于评估，不可作为训练模式",
    },
    "activation_type": {
        "relu": "ReLU",
        "pwl": "分段线性（knee / tail_slope）",
        "tanh": "Tanh（beta 控制饱和度）",
    },
    "noise_mode": {
        "local_relative": "按局部相对量缩放噪声",
        "full_scale": "按满量程缩放噪声",
    },
    "tia_saturation_mode": {
        "absolute": "绝对饱和值",
        "full_scale": "满量程比例",
    },
    "activation_threshold_mode": {
        "absolute": "绝对阈值",
        "full_scale": "满量程比例",
    },
    "domain": {
        "shape": "尺度无关（推荐）",
        "absolute": "绝对权重域",
    },
}


FIELD_LABELS = {
    "epochs": "训练轮数",
    "learning_rate": "学习率",
    "scheduler": "学习率调度器",
    "scheduler_step_epochs": "Step 调度间隔",
    "scheduler_gamma": "学习率衰减倍率",
    "scheduler_milestones": "MultiStep 里程碑",
    "scheduler_patience": "Plateau 等待轮数",
    "cosine_restart_epochs": "Cosine 重启周期",
    "gradient_clip_norm": "梯度裁剪上限",
    "seed": "随机种子",
    "optimization_batch_pixels": "参数更新像素微批大小",
    "train_pixels_per_image": "每张图每轮抽样像素数",
    "images_per_batch": "图像批大小",
    "loader_workers": "数据加载进程数",
    "resume": "断点恢复方式",
    "initial_checkpoint": "初始权重检查点",
    "target_algorithm": "监督目标算法",
    "error_function": "误差函数",
    "loss.error_function": "误差函数",
    "loss.terms.iq.weight": "IQ 损失系数",
    "loss.terms.envelope.weight": "包络损失系数",
    "loss.charbonnier_epsilon": "Charbonnier 平滑项",
    "loss.envelope_epsilon": "包络数值稳定项",
    "loss.terms.unity.weight": "无失真软约束系数",
    "distillation_weight": "蒸馏系数",
    "distillation_checkpoint": "蒸馏教师检查点",
    "input_normalization": "输入归一化",
    "hidden_width": "隐藏层宽度",
    "hidden_layers": "隐藏层数量",
    "dropout": "Dropout",
    "branch_mode": "单双支路",
    "centering": "样本中心化",
    "centering_layers": "中心化作用层",
    "normalization": "隐藏特征归一化",
    "normalization_layers": "归一化作用层",
    "weight_transform": "权重处理",
    "weight_transform_layers": "WS 作用层",
    "weight_transform_epsilon": "WS epsilon",
    "running_stat_momentum": "运行统计动量",
    "running_stat_epsilon": "运行统计 epsilon",
    "batch_renorm_rmax": "Batch Renorm r 上限",
    "batch_renorm_dmax": "Batch Renorm d 上限",
    "layer_activations": "逐层激活函数",
    "output_weights": "实/复权重",
    "output_controls": "原生输出控制点 K",
    "projection_controls": "后置投影控制点 Kp",
    "output_interpolation": "控制点插值方式",
    "interpolation_bits": "插值定点位宽",
    "output_domain": "输出域",
    "unity_constraint": "Unity 约束",
    "input_tile_features": "FC1 实特征分块大小",
    "dynamic_aperture": "动态孔径",
    "f_number": "F-Number",
    "h5_files": "训练 H5 列表",
    "train_ratio": "训练比例",
    "validation_ratio": "验证比例",
    "test_ratio": "测试比例",
    "split_file": "固定划分 CSV",
    "make_new_split": "重新生成划分",
    "use_train_as_validation": "训练集兼作验证集",
    "output_directory": "训练结果目录",
    "use_tgc": "训练期预览使用 TGC",
    "save_images_every_epochs": "预览图保存间隔",
    "visualization_acquisition": "预览采集标识",
    "mode": "运行模式",
    "inter_layer": "层间数据通路",
    "inter_layer_bits": "层间位宽",
    "weight_bits": "忆阻器权重位宽",
    "input_bits": "输入 DAC 位宽",
    "control_bits": "控制 ADC 位宽",
    "bias_bits": "普通 bias 位宽",
    "g_min": "最小电导",
    "g_max": "最大电导",
    "per_output_channel": "按输出通道校准",
    "observer_enabled": "启用 observer",
    "fixed_observer_calibration": "固定 observer 校准",
    "observer_calibration_epochs": "动态 observer 校准轮数",
    "ptq_calibration_samples": "PTQ 校准帧数",
    "public_gain_max": "公开增益显示上限",
    "active_profile": "当前非理想 profile",
    "knee": "PWL 转折点",
    "tail_slope": "PWL 尾部斜率",
    "beta": "Tanh beta",
    "loss.terms.d1.domain": "D1 作用域",
    "loss.terms.d1.weight": "D1 正则系数",
    "loss.terms.d2.domain": "D2 作用域",
    "loss.terms.d2.weight": "D2 正则系数",
    "loss.terms.lrange.limit": "raw control 范围阈值",
    "loss.terms.lrange.weight": "raw control 范围正则系数",
}


# 这些路径对应当前 train/config.yaml 的真实层级。
# 每节内先列公共参数，再列互斥选择及其子参数（大参数在前，小参数在后）；
# 子参数是否询问由 FIELD_ACTIVE 的条件决定。
QUICK_FIELDS = {
    "training": [
        "training.epochs",
        "training.learning_rate",
        "training.seed",
        "training.gradient_clip_norm",
        "training.optimization_batch_pixels",
        "training.train_pixels_per_image",
        "training.images_per_batch",
        "training.loader_workers",
        "training.scheduler",
        "training.scheduler_step_epochs",
        "training.scheduler_gamma",
        "training.scheduler_milestones",
        "training.scheduler_patience",
        "training.cosine_restart_epochs",
        "training.resume",
        "training.initial_checkpoint",
    ],
    "run": ["run.mode"],
    "input": ["input.input_normalization"],
    "data": [
        "data.h5_files",
        "data.train_ratio",
        "data.validation_ratio",
        "data.test_ratio",
        "data.split_file",
        "data.make_new_split",
        "data.use_train_as_validation",
    ],
    "backbone": [
        "backbone.hidden_width",
        "backbone.hidden_layers",
        "backbone.dropout",
        "backbone.branch_mode",
        "backbone.centering",
        "backbone.centering_layers",
        "backbone.normalization",
        "backbone.normalization_layers",
        "backbone.weight_transform",
        "backbone.weight_transform_layers",
        "backbone.weight_transform_epsilon",
        "backbone.running_stat_momentum",
        "backbone.running_stat_epsilon",
        "backbone.batch_renorm_rmax",
        "backbone.batch_renorm_dmax",
    ],
    "output_head": [
        "output_head.output_weights",
        "output_head.output_controls",
        "output_head.projection_controls",
        "output_head.output_interpolation",
        "output_head.output_domain",
        "output_head.unity_constraint",
    ],
    "target": [
        "target.target_algorithm",
    ],
    "loss": [
        "loss.error_function",
        "loss.charbonnier_epsilon",
        "loss.envelope_epsilon",
        "loss.terms.iq.weight",
        "loss.terms.envelope.weight",
        "loss.terms.unity.weight",
        "loss.terms.d1.weight",
        "loss.terms.d1.domain",
        "loss.terms.d2.weight",
        "loss.terms.d2.domain",
        "loss.terms.lrange.weight",
        "loss.terms.lrange.limit",
    ],
    "distillation": ["distillation.distillation_weight", "distillation.distillation_checkpoint"],
    "aperture": ["aperture.dynamic_aperture", "aperture.f_number"],
    "output": [
        "output.output_directory",
        "output.use_tgc",
        "output.save_images_every_epochs",
        "output.visualization_acquisition",
    ],
}


# 互斥选择下的小参数只在父选择适用时才询问；键为完整配置路径。
FIELD_ACTIVE = {
    "training.scheduler_step_epochs": lambda c: _lookup(c, "training.scheduler") == "step",
    "training.scheduler_gamma": lambda c: _lookup(c, "training.scheduler") in {"step", "multistep"},
    "training.scheduler_milestones": lambda c: _lookup(c, "training.scheduler") == "multistep",
    "training.scheduler_patience": lambda c: _lookup(c, "training.scheduler") == "plateau",
    "training.cosine_restart_epochs": lambda c: _lookup(c, "training.scheduler") == "cosine",
    "training.initial_checkpoint": lambda c: _lookup(c, "training.resume") != "all",
    "loss.terms.unity.weight": lambda c: _lookup(c, "output_head.unity_constraint") == "free",
    "loss.charbonnier_epsilon": lambda c: _lookup(c, "loss.error_function") == "charbonnier",
    "loss.terms.d1.domain": lambda c: float(_lookup(c, "loss.terms.d1.weight") or 0) > 0,
    "loss.terms.d2.domain": lambda c: float(_lookup(c, "loss.terms.d2.weight") or 0) > 0,
    "loss.terms.lrange.limit": lambda c: float(_lookup(c, "loss.terms.lrange.weight") or 0) > 0,
    "distillation.distillation_checkpoint": lambda c: float(_lookup(c, "distillation.distillation_weight") or 0) > 0,
    "backbone.normalization_layers": lambda c: _lookup(c, "backbone.normalization") != "none",
    "backbone.weight_transform_layers": lambda c: _lookup(c, "backbone.weight_transform") == "ws",
    "backbone.weight_transform_epsilon": lambda c: _lookup(c, "backbone.weight_transform") == "ws",
    "backbone.centering_layers": lambda c: _lookup(c, "backbone.centering") == "sample_mean",
    "backbone.running_stat_momentum": lambda c: _lookup(c, "backbone.normalization") in {"running_zscore", "batch_renorm"},
    "backbone.running_stat_epsilon": lambda c: _lookup(c, "backbone.normalization") in {"running_zscore", "batch_renorm"},
    "backbone.batch_renorm_rmax": lambda c: _lookup(c, "backbone.normalization") == "batch_renorm",
    "backbone.batch_renorm_dmax": lambda c: _lookup(c, "backbone.normalization") == "batch_renorm",
    "output_head.projection_controls": lambda c: int(_lookup(c, "output_head.output_controls") or 0) == 0,
    "output_head.output_interpolation": lambda c: (
        int(_lookup(c, "output_head.output_controls") or 0) > 0
        or int(_lookup(c, "output_head.projection_controls") or 0) > 0
    ),
}


def _lookup(config: dict, path: str):
    try:
        return get_config_value(config, path)
    except (KeyError, TypeError):
        return None


def field_active(path: str, config: dict) -> bool:
    rule = FIELD_ACTIVE.get(path)
    if rule is None:
        return True
    return bool(rule(config))


def sync_dependent_fields(config: dict) -> None:
    """保留用户填写的语义字段，由统一校验器报告非法组合。"""
    return
SECTION_LABELS = {
    "training": "训练过程",
    "target": "监督目标与损失",
    "distillation": "蒸馏",
    "input": "输入表示",
    "activation_functions": "激活函数库",
    "backbone": "网络骨干",
    "output_head": "输出权重",
    "loss": "损失",
    "aperture": "动态孔径",
    "data": "数据划分",
    "output": "输出与预览",
    "run": "运行模式",
}


def title(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def explain(enabled: bool, key: str) -> None:
    if enabled and key in HELP:
        print(f"\n说明：{HELP[key]}")


def read_line(prompt: str) -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        raise ExitCommandError from None
    low = value.lower()
    if low in {"b", "back", "返回", "上一步"}:
        raise BackCommandError
    if low in {"q", "quit", "exit", "退出"}:
        raise ExitCommandError
    return value


def ask_text(prompt: str, default=None) -> str:
    suffix = "" if default is None else f"（默认：{default}）"
    value = read_line(f"{prompt}{suffix}: ")
    return str(default) if value == "" and default is not None else value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    while True:
        raw = read_line(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").lower()
        if raw == "":
            return default
        if raw in {"y", "yes", "是", "1", "true"}:
            return True
        if raw in {"n", "no", "否", "0", "false"}:
            return False
        print("请输入 y 或 n。")


def ask_choice(prompt: str, options: list[tuple[str, str]], default: str) -> str:
    if not options:
        raise ValueError("没有可选项")
    print(f"\n{prompt}")
    for index, (key, description) in enumerate(options, 1):
        mark = "   ← 默认" if key == default else ""
        print(f"  {index}. {description}{mark}")
    while True:
        raw = read_line("请输入序号或名称: ")
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        for key, _ in options:
            if raw.lower() == str(key).lower():
                return key
        print("选择无效，请重新输入。")


def ask_int(prompt: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = ask_text(prompt, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            print("请输入整数。")
            continue
        if minimum is not None and value < minimum:
            print(f"数值必须大于等于 {minimum}。")
            continue
        return value


def ask_optional_int(
    prompt: str,
    minimum: int = 0,
    maximum: int | None = None,
    choices: tuple[int, ...] | None = None,
) -> str:
    while True:
        raw = ask_text(prompt, "").strip()
        if not raw:
            return ""
        try:
            value = int(raw)
        except ValueError:
            print("请输入整数，留空表示使用配置值。")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            print(f"数值必须位于 [{minimum}, {maximum}]。" if maximum is not None else f"数值必须大于等于 {minimum}。")
            continue
        if choices is not None and value not in choices:
            print(f"可选值为：{', '.join(str(item) for item in choices)}。")
            continue
        return str(value)


def parse_yaml_value(raw: str, expected):
    """Parse a YAML scalar/list/mapping while retaining the current type."""
    value = yaml.safe_load(raw)
    if expected is None:
        return value
    if isinstance(expected, bool):
        if not isinstance(value, bool):
            raise ValueError("请输入 true 或 false")
        return value
    if isinstance(expected, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("请输入整数")
        return value
    if isinstance(expected, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("请输入数字")
        if not math.isfinite(float(value)):
            raise ValueError("请输入有限数字")
        return float(value)
    if isinstance(expected, str):
        if not isinstance(value, str):
            raise ValueError("请输入文本")
        return value
    if isinstance(expected, list):
        if not isinstance(value, list):
            raise ValueError("请输入 YAML 列表，例如 [fc2] 或 [393]")
        return value
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            raise ValueError("请输入 YAML 映射，例如 {fc2: relu}")
        return value
    return value


def _path_key(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def field_label(path: str) -> str:
    return FIELD_LABELS.get(path) or FIELD_LABELS.get(_path_key(path)) or _path_key(path)


def available_target_algorithms(config: dict) -> list[str]:
    """读取所有训练 H5 的共同复数 IQ 监督目标。"""
    data = config.get("data", {})
    h5_files = data.get("h5_files", []) if isinstance(data, dict) else []
    if not isinstance(h5_files, list) or not h5_files:
        return []
    import h5py

    target_sets = []
    for raw_path in h5_files:
        text = str(raw_path)
        candidates = [
            resolve_path(text, ACTIVE_CONFIG_BASE_DIR),
            resolve_path(text, ROOT),
            resolve_path(text, TRAIN_DIR),
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"训练 H5 不存在: {text}")
        with h5py.File(path, "r") as handle:
            if "config_yaml" not in handle:
                raise KeyError(f"训练 H5 缺少 config_yaml: {path}")
            raw = handle["config_yaml"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = yaml.safe_load(raw) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"训练 H5 config_yaml 必须是映射: {path}")
        generation = metadata.get("generation", {})
        targets = generation.get("training_targets", {}).get("complex_iq_targets", {}) if isinstance(generation, dict) else {}
        if not isinstance(targets, dict):
            raise ValueError(f"训练 H5 缺少 generation.training_targets.complex_iq_targets: {path}")
        target_sets.append({str(name) for name in targets})
    return sorted(set.intersection(*target_sets)) if target_sets else []


def _choice_options(path: str, current, config: dict | None = None) -> list[tuple[str, str]] | None:
    if path == "run.mode":
        # ptq is an evaluation type, not a valid mban.py training mode.
        key = "run.mode"
        values = ("software", "qat")
    elif path == "target.target_algorithm":
        values = tuple(available_target_algorithms(config or {}))
        if not values:
            return None
        key = "target_algorithm"
    elif ".layer_activations." in path and isinstance(current, str):
        key = "activation_type"
        values = CHOICES["activation_type"]
    else:
        key = path
        if key not in CHOICES:
            key = _path_key(path)
        if key not in CHOICES:
            for candidate_key in CHOICES:
                if path.endswith("." + candidate_key):
                    key = candidate_key
                    break
            else:
                if key.endswith("noise_mode"):
                    key = "noise_mode"
                elif key.endswith("saturation_mode"):
                    key = "tia_saturation_mode"
                elif key.endswith("threshold_mode"):
                    key = "activation_threshold_mode"
        values = CHOICES.get(key)
    if path == "backbone.normalization" and _lookup(config or {}, "backbone.branch_mode") == "single":
        values = tuple(value for value in values or () if value != "l1")
    if path == "output_head.output_domain" and _lookup(config or {}, "output_head.output_weights") == "complex":
        values = tuple(value for value in values or () if value != "nonnegative")
    if values is None:
        return None
    descriptions = CHOICE_DESCRIPTIONS.get(key, {})
    return [(value, descriptions.get(value, value)) for value in values]


def help_key(path: str) -> str:
    """定位字段对应的教学说明：整条路径 → 末段 → 去掉顶层段前缀。"""
    if path in HELP:
        return path
    key = _path_key(path)
    if key in HELP:
        return key
    for candidate in HELP:
        if path.endswith("." + candidate):
            return candidate
    section = path.split(".", 1)[0]
    if section in HELP:
        return section
    return path


def validate_interactive_value(path: str, value, config: dict) -> None:
    if path in {"training.epochs", "training.scheduler_step_epochs", "training.scheduler_patience"} and value < 0:
        raise ValueError("数值必须大于等于 0")
    if path == "training.scheduler_step_epochs" and value <= 0:
        raise ValueError("数值必须大于 0")
    if path == "training.learning_rate" and value <= 0:
        raise ValueError("学习率必须大于 0")
    if path == "training.scheduler_gamma" and not 0 < value <= 1:
        raise ValueError("衰减倍率必须位于 (0, 1]")
    if path in {"training.optimization_batch_pixels", "training.images_per_batch"} and value <= 0:
        raise ValueError("数值必须大于 0")
    if path in {"training.train_pixels_per_image", "training.gradient_clip_norm"} and value < 0:
        raise ValueError("数值必须大于等于 0")
    if path == "training.loader_workers" and value < -1:
        raise ValueError("数据加载进程数必须为 -1 或非负整数")
    if (
        path
        in {
            "loss.terms.iq.weight",
            "loss.terms.envelope.weight",
            "loss.terms.unity.weight",
            "loss.terms.d1.weight",
            "loss.terms.d2.weight",
            "loss.terms.lrange.weight",
            "loss.terms.lrange.limit",
            "loss.envelope_epsilon",
            "loss.charbonnier_epsilon",
        }
        and value < 0
    ):
        raise ValueError("损失参数不能为负")
    if path in {"loss.envelope_epsilon", "loss.charbonnier_epsilon"} and value <= 0:
        raise ValueError("数值必须大于 0")
    if path == "backbone.hidden_width" and value < 0:
        raise ValueError("隐藏层宽度必须大于等于 0；0 表示自动推导")
    if path == "backbone.hidden_layers" and value < 1:
        raise ValueError("隐藏层数量必须大于等于 1")
    if path == "backbone.dropout" and not 0 <= value < 1:
        raise ValueError("Dropout 必须位于 [0, 1)")
    if path in {"backbone.running_stat_momentum", "backbone.running_stat_epsilon"} and value <= 0:
        raise ValueError("运行统计参数必须大于 0")
    if path == "backbone.batch_renorm_rmax" and value < 1:
        raise ValueError("Batch Renorm r 上限必须大于等于 1")
    if path == "backbone.batch_renorm_dmax" and value < 0:
        raise ValueError("Batch Renorm d 上限必须大于等于 0")
    if path in {"output_head.output_controls", "output_head.projection_controls"} and value < 0:
        raise ValueError("控制点数量必须大于等于 0")
    if path == "output_head.output_controls" and value == 1:
        raise ValueError("output_controls 必须为 0 或不小于 2")
    if path == "aperture.f_number" and value <= 0:
        raise ValueError("F-Number 必须大于 0")
    if path in {"data.train_ratio", "data.validation_ratio", "data.test_ratio"} and not 0 <= value <= 1:
        raise ValueError("数据比例必须位于 [0, 1]")
    if path == "output.save_images_every_epochs" and value < 0:
        raise ValueError("保存间隔必须大于等于 0")


def ask_field(path: str, current, teaching: bool, config: dict):
    explain(teaching, help_key(path))
    label = field_label(path)
    options = _choice_options(path, current, config)
    if options:
        default = str(current) if str(current) in {item[0] for item in options} else options[0][0]
        selected = ask_choice(f"{label}（{path}）", options, default)
        if isinstance(current, int) and not isinstance(current, bool):
            selected = int(selected)
        validate_interactive_value(path, selected, config)
        return selected
    if isinstance(current, bool):
        return ask_yes_no(f"{label}（{path}）", current)
    if isinstance(current, list) and path in LIST_ITEM_KINDS:
        return ask_list(label, path, current, LIST_ITEM_KINDS[path])
    shown = (
        "null"
        if current is None
        else yaml.safe_dump(
            current,
            allow_unicode=True,
            default_flow_style=True,
            sort_keys=False,
        ).strip()
    )
    while True:
        raw = ask_text(f"{label}（{path}）", shown)
        try:
            value = parse_yaml_value(raw, current)
            validate_interactive_value(path, value, config)
            return value
        except (ValueError, yaml.YAMLError) as exc:
            print(f"输入无效：{exc}")


LIST_ITEM_KINDS = {
    "data.h5_files": "path",
    "backbone.centering_layers": "layer",
    "backbone.normalization_layers": "layer",
    "backbone.weight_transform_layers": "layer",
    "training.scheduler_milestones": "int",
}


def _render_list_item(kind: str, item) -> str:
    return str(item)


def _parse_list_item(kind: str, raw: str):
    if kind == "path":
        value = raw.strip().strip('"').strip("'")
        if not value:
            raise ValueError("路径不能为空")
        return value
    if kind == "layer":
        value = raw.strip().strip('"').strip("'")
        if not value.startswith("fc"):
            raise ValueError("层名应为 fc1、fc2 等格式")
        return value
    if kind == "int":
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError("请输入整数") from None
        if value <= 0:
            raise ValueError("数值必须大于 0")
        return value
    raise ValueError(f"未知的列表类型：{kind}")


LIST_HINTS = {
    "path": "输入新路径追加；输入序号删除（可逗号分隔多个）；留空确认。",
    "layer": "输入层名（如 fc3）追加；输入序号删除（可逗号分隔多个）；留空确认。",
    "int": "输入整数追加（超出序号范围的数字视为新值）；输入序号删除；留空确认。",
}


def ask_list(prompt: str, path: str, current, kind: str) -> list:
    """逐项编辑列表：留空确认、序号删除、新值追加，无需手写 YAML。"""
    items = list(current)
    if items:
        default_text = "、".join(_render_list_item(kind, item) for item in items)
        print(f"\n{prompt}（{path}）（当前 {len(items)} 项，默认：{default_text}）")
    else:
        print(f"\n{prompt}（{path}）（当前为空）")
    for index, item in enumerate(items, 1):
        print(f"  {index}. {_render_list_item(kind, item)}")
    print(LIST_HINTS[kind])
    while True:
        raw = read_line("请输入: ").strip()
        if raw == "":
            return items
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            continue
        if kind == "int" and len(parts) == 1 and parts[0].isdigit():
            index = int(parts[0])
            if 1 <= index <= len(items):
                removed = items.pop(index - 1)
                print(f"  已删除：{_render_list_item(kind, removed)}")
                continue
            # 超出范围的整数：当作新值追加，而不是报序号无效。
        elif all(part.isdigit() for part in parts):
            indices = [int(part) - 1 for part in parts]
            invalid = [idx + 1 for idx in indices if not 0 <= idx < len(items)]
            if invalid:
                print(f"序号无效：{', '.join(str(value) for value in invalid)}")
                continue
            for idx in sorted(set(indices), reverse=True):
                removed = items.pop(idx)
                print(f"  已删除：{_render_list_item(kind, removed)}")
            continue
        try:
            parsed = _parse_list_item(kind, raw)
        except ValueError as exc:
            print(exc)
            continue
        items.append(parsed)
        print(f"  已添加：{_render_list_item(kind, parsed)}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} 不是 YAML 映射")
    return value


def hardware_eval_path(config_path: Path) -> Path:
    return resolve_hardware_config(config_path) or DEFAULT_HARDWARE_CONFIG


def display_command(command: list[str], gpu: str | None = None) -> str:
    quoted = " ".join(shlex.quote(str(part)) for part in command)
    prefix = "" if gpu is None else f"$env:CUDA_VISIBLE_DEVICES='{gpu}'; "
    return prefix + quoted


def get_config_value(config: dict, path: str):
    value = config
    for part in path.split("."):
        value = value[part]
    return value


def set_config_value(config: dict, path: str, value) -> None:
    parts = path.split(".")
    parent = config
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value


def normalize_training_paths(config: dict, base_config_path: Path) -> None:
    """Make path fields absolute so generated YAML can live anywhere."""
    base = base_config_path.resolve().parent
    data = config.get("data", {})
    if isinstance(data.get("h5_files"), list):
        data["h5_files"] = [str(resolve_path(path, base)) for path in data["h5_files"]]
    for section, key in (
        ("data", "split_file"),
        ("training", "initial_checkpoint"),
        ("distillation", "distillation_checkpoint"),
        ("output", "output_directory"),
    ):
        section_data = config.get(section, {})
        if isinstance(section_data, dict) and section_data.get(key) is not None:
            section_data[key] = str(resolve_path(section_data[key], base))


def validate_training_inputs(config: dict, base_config_path: Path, generated_path: Path) -> list[str]:
    """检查训练入口会直接使用的文件和输出状态，避免保存后才失败。"""
    notes: list[str] = []
    if generated_path.resolve() == base_config_path.resolve():
        raise ValueError("生成配置不能覆盖基础训练配置")
    data = config.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("data 必须是映射")
    h5_files = data.get("h5_files", [])
    if not isinstance(h5_files, list) or not h5_files:
        raise ValueError("至少选择一个训练 H5")
    for raw_path in h5_files:
        path = resolve_path(raw_path)
        if not path.is_file():
            raise ValueError(f"训练 H5 不存在：{path}")
        certificate = Path(f"{path}.check.json")
        if not certificate.is_file():
            raise ValueError(f"H5 缺少全量检查凭证：{certificate}；请先运行 data/checks/check_data.py --full")
    split_file = data.get("split_file")
    if split_file and not resolve_path(split_file).is_file():
        raise ValueError(f"固定划分 CSV 不存在：{resolve_path(split_file)}")

    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training 必须是映射")
    for section, key in (("training", "initial_checkpoint"), ("distillation", "distillation_checkpoint")):
        value = config.get(section, {}).get(key) if isinstance(config.get(section), dict) else None
        if value and not resolve_path(value).is_file():
            raise ValueError(f"{key} 不存在：{resolve_path(value)}")

    output_value = config.get("output", {}).get("output_directory")
    if not output_value:
        raise ValueError("必须指定 output.output_directory")
    output = resolve_path(output_value)
    if output.exists() and not output.is_dir():
        raise ValueError(f"训练输出路径不是目录：{output}")
    latest = output / "latest.pth"
    resume = training.get("resume")
    if resume == "none" and latest.is_file():
        raise ValueError("resume=none 不能复用含 latest.pth 的输出目录；请换新目录或选择 weights/all")
    if resume == "all" and not latest.is_file():
        notes.append("resume=all 但输出目录没有 latest.pth，训练入口将从头初始化")
    if resume == "weights" and not training.get("initial_checkpoint") and not latest.is_file():
        notes.append("resume=weights 未提供 initial_checkpoint 且没有 latest.pth，将等同于随机初始化")

    ratios = [data.get("train_ratio"), data.get("validation_ratio"), data.get("test_ratio")]
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in ratios) and not math.isclose(
        sum(ratios), 1.0, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise ValueError("train/validation/test 比例之和必须为 1")
    return notes


def enforce_training_constraints(
    config: dict,
    config_path: Path | None = None,
    hardware_config_path: Path | None = None,
) -> list[str]:
    """Validate using the same loader as ``mban.py``.

    A few train-loop constraints depend on the H5-derived network width and are
    intentionally left to ``mban_core.training.train``. Static YAML semantics
    must, however, be checked before the generated file is accepted.
    """
    output_head = config.get("output_head", {})
    run = config.get("run", {})
    if run.get("mode") == "ptq":
        raise ValueError("run.mode=ptq 只用于 evaluate_mban.py 评估，不能作为 mban.py 训练模式")
    output_controls = int(output_head.get("output_controls", 0))
    projection_controls = int(output_head.get("projection_controls", 0))
    if output_controls > 0 and projection_controls > 0:
        raise ValueError("output_controls 与 projection_controls 不能同时启用；Direct+Proj 请将 output_controls 设为 0")
    if output_controls == 1:
        raise ValueError("output_controls 必须为 0 或不小于 2")
    validation_path = Path(config_path or DEFAULT_TRAIN_CONFIG).resolve()
    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))
    try:
        from mban_core.config import load_config
    except ImportError as exc:
        raise RuntimeError("无法导入 train/mban_core/config.py，无法复用训练配置校验") from exc
    load_config(validation_path, raw_config=config, hardware_config_path=hardware_config_path)
    return []


def build_field_sections(config: dict) -> dict[str, list[str]]:
    fields = {section: list(paths) for section, paths in QUICK_FIELDS.items()}
    functions = config.get("activation_functions", {})
    for function_name, function_config in functions.items():
        if isinstance(function_config, dict):
            fields.setdefault("activation_functions", []).extend(
                f"activation_functions.{function_name}.{parameter}" for parameter in function_config
            )
    layers = config.get("backbone", {}).get("layer_activations", {})
    if isinstance(layers, dict):
        fields.setdefault("layer_activations", []).extend(
            f"backbone.layer_activations.{layer}" for layer in sorted(layers)
        )
    return fields


def normalize_gpu_selection(raw: str) -> tuple[str, str]:
    value = raw.strip()
    if value.lower() in {"", "cpu", "none", "无"}:
        return "", "cpu"
    if any(char.isspace() for char in value):
        raise ValueError("GPU 编号不能包含空格")
    return value, "cuda"


def ask_device() -> tuple[str, str]:
    while True:
        raw = ask_text("使用的 GPU 编号；输入 CPU 留空", "1")
        try:
            return normalize_gpu_selection(raw)
        except ValueError as exc:
            print(exc)


def run_command(command: list[str], gpu: str | None) -> int:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return result.returncode


def default_generated_config_path(timestamp: str) -> Path:
    return TRAIN_DIR / "generated_configs" / f"train_{timestamp}.yaml"


def run_training_wizard(teaching: bool, execute_default: bool) -> None:
    global ACTIVE_CONFIG_BASE_DIR
    title("MBAN 训练配置向导")
    explain(teaching, "base")
    base_path = resolve_path(ask_text("基础训练配置", str(DEFAULT_TRAIN_CONFIG)))
    ACTIVE_CONFIG_BASE_DIR = base_path.resolve().parent
    config = deepcopy(load_yaml(base_path))
    for section in ("hardware", "hardware_execution", "nonidealities", "qat_schedule"):
        config.pop(section, None)
    if isinstance(config.get("output_head"), dict):
        config["output_head"].pop("interpolation_bits", None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_path = resolve_path(ask_text("生成配置保存路径", str(default_generated_config_path(timestamp))))

    # 向导默认创建独立实验目录，避免接受 config.yaml 中的历史输出目录后覆盖结果。
    config.setdefault("output", {})["output_directory"] = str(DEFAULT_RESULTS_DIR / f"wizard_{timestamp}")
    # 新输出目录没有 latest.pth，默认从头开始；需要续训时由用户明确选择 all/weights。
    config.setdefault("training", {})["resume"] = "none"
    config["training"]["initial_checkpoint"] = None
    sync_dependent_fields(config)
    section_fields = build_field_sections(config)
    field_entries = [(section, path) for section, paths in section_fields.items() for path in paths]
    field_index = 0
    current_section = None
    while field_index < len(field_entries):
        section, path = field_entries[field_index]
        if section != current_section:
            title(SECTION_LABELS.get(section, section))
            current_section = section
        if not field_active(path, config):
            field_index += 1
            continue
        try:
            current = get_config_value(config, path)
        except (KeyError, TypeError):
            field_index += 1
            continue
        try:
            value = ask_field(path, current, teaching, config)
        except BackCommandError:
            previous_index = field_index - 1
            while previous_index >= 0:
                _, previous_path = field_entries[previous_index]
                try:
                    get_config_value(config, previous_path)
                except (KeyError, TypeError):
                    previous_index -= 1
                    continue
                if field_active(previous_path, config):
                    break
                previous_index -= 1
            if previous_index < 0:
                raise
            print(f"返回上一项：{field_entries[previous_index][1]}")
            field_index = previous_index
            current_section = None
            continue
        set_config_value(config, path, value)
        sync_dependent_fields(config)
        field_index += 1

    normalize_training_paths(config, base_path)
    try:
        notes = validate_training_inputs(config, base_path, generated_path)
        notes.extend(
            enforce_training_constraints(
                config,
                generated_path,
                hardware_eval_path(base_path),
            )
        )
    except (ValueError, RuntimeError) as exc:
        print(f"配置校验失败：{exc}")
        print("未保存配置，也未启动训练。请重新运行向导并修正相关字段。")
        return

    gpu, _ = ask_device()
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(generated_path),
        "--eval-config",
        str(hardware_eval_path(base_path)),
    ]
    title("训练配置确认")
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    for note in notes:
        print(f"自动修正：{note}")
    print("命令：")
    print(display_command(command, gpu))
    execute = ask_yes_no("保存配置后立即开始训练", execute_default)
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    with generated_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    print(f"配置已保存：{generated_path}")
    if execute:
        return_code = run_command(command, gpu)
        if return_code:
            raise SystemExit(f"训练失败，退出码 {return_code}")


def ask_many(prompt: str, options: list[tuple[str, str]], defaults: list[str]) -> list[str]:
    keys = [key for key, _ in options]
    default_text = ",".join(str(keys.index(key) + 1) for key in defaults)
    print(f"\n{prompt}")
    for index, (_, description) in enumerate(options, 1):
        print(f"  {index}. {description}")
    while True:
        raw = ask_text("输入序号或名称，多个用逗号分隔", default_text)
        if raw.strip().lower() == "all":
            return keys
        selected = []
        try:
            for token in raw.split(","):
                token = token.strip()
                if not token:
                    continue
                if token.isdigit():
                    key = keys[int(token) - 1]
                else:
                    key = next(key for key in keys if key.lower() == token.lower())
                if key in selected:
                    raise ValueError(f"重复选择：{key}")
                selected.append(key)
            if selected:
                return selected
        except (IndexError, StopIteration, ValueError) as error:
            print(f"选择无效：{error}")


def _eval_item_name(item: dict, index: int) -> str:
    return str(item.get("display_name") or f"model{index}")


def _make_eval_plan(config: dict, model_indices: list[str], evaluations: list[str], test_frames: int) -> dict:
    plan = deepcopy(config)
    items = plan["models"]["items"]
    plan["models"]["items"] = [items[int(index) - 1] for index in model_indices]
    plan["enabled"] = {"evaluations": evaluations}
    if "invivo" in evaluations:
        plan["evaluations"]["invivo"]["test_frames"] = test_frames
    return plan


def _absolute_eval_plan(plan: dict, base_path: Path) -> dict:
    plan = deepcopy(plan)
    base = base_path.resolve().parent
    for key, value in plan["paths"].items():
        plan["paths"][key] = str(resolve_path(value, base))
    for item in plan["models"]["items"]:
        item["checkpoint"] = str(resolve_path(item["checkpoint"], base))
    for item in plan["evaluations"].values():
        for key in ("input", "references_json"):
            if key in item:
                item[key] = str(resolve_path(item[key], base))
    return plan


def write_selected_eval(plan: dict) -> Path:
    SELECTED_EVAL.parent.mkdir(parents=True, exist_ok=True)
    selected = _absolute_eval_plan(plan, DEFAULT_EVAL)
    SELECTED_EVAL.write_text(
        "# 由 mban_wizard_cn.py 生成；详细配置见 train/eval.yaml。\n"
        + yaml.safe_dump(selected, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    load_eval(SELECTED_EVAL)
    return SELECTED_EVAL


def run_validation_wizard(teaching: bool, execute_default: bool) -> None:
    del teaching
    title("MBAN 评估向导")
    config = load_yaml(DEFAULT_EVAL)
    model_items = config["models"]["items"]
    model_options = [
        (str(index), f"{_eval_item_name(item, index)}：{item.get('checkpoint', '')}")
        for index, item in enumerate(model_items, 1)
    ]
    selected_indices = ask_many("选择模型（同一计划统一使用一个评估类型）", model_options, [str(index) for index in range(1, len(model_items) + 1)])
    model_type = ask_choice(
        "选择模型评估类型",
        [("fp32", "FP32：软件浮点"), ("qat", "QAT：读取 checkpoint 硬件状态"), ("ptq", "PTQ：评估时量化并校准")],
        str(config["models"].get("model_type", "fp32")),
    )
    evaluation_items = config["evaluations"]
    descriptions = {
        "scenes": "四场景",
        "invivo": "活体固定帧",
        "scenes_mc4": "四场景 MC4",
        "invivo_mc4": "活体 MC4",
        "weights": "权重与控制量诊断",
        "figures": "评估图表",
        "trace_qat_ptq": "硬件链路 trace",
    }
    eval_options = [
        (name, f"{descriptions.get(name, name)}（{item['mode']}）") for name, item in evaluation_items.items()
    ]
    defaults = list(config["enabled"]["evaluations"])
    selected_evaluations = ask_many("选择评估内容", eval_options, defaults)
    test_frames = int(evaluation_items.get("invivo", {}).get("test_frames", 50))
    if "invivo" in selected_evaluations:
        test_frames = ask_int("活体帧数", test_frames, minimum=1)
    plan = _make_eval_plan(config, selected_indices, selected_evaluations, test_frames)
    plan["models"]["model_type"] = model_type
    plan_path = write_selected_eval(plan)
    print(f"\n模型：{', '.join(_eval_item_name(model_items[int(index) - 1], int(index)) for index in selected_indices)}")
    print(f"类型：{model_type}")
    print(f"评估：{', '.join(selected_evaluations)}")
    print(f"计划：{plan_path}")
    print(f"命令：{display_command([sys.executable, str(VALIDATE_SCRIPT), '--eval', str(plan_path)])}")
    if ask_yes_no("立即开始评估", execute_default):
        return_code = run_command([sys.executable, str(VALIDATE_SCRIPT), "--eval", str(plan_path)], None)
        if return_code:
            raise SystemExit(f"评估失败，退出码 {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MBAN 训练与评估中文向导")
    parser.add_argument("--workflow", choices=("train", "validate"))
    parser.add_argument("--execute", action="store_true", help="把最终执行问题的默认答案设为是")
    cli = parser.parse_args()
    while True:
        title("MBAN 训练与评估中文向导")
        print("输入 q 可随时退出；训练参数中输入 b 返回上一项，其他位置输入 b 取消当前工作流。")
        try:
            if cli.workflow == "validate":
                teaching = False
                workflow = "validate"
            else:
                teaching = ask_yes_no("开启参数教学说明", True)
                explain(teaching, "workflow")
                workflow = cli.workflow or ask_choice(
                    "请选择工作流",
                    [
                        ("train", "训练：生成独立 YAML 并启动 mban.py"),
                        ("validate", "评估：调用 evaluate_mban.py 的统一模式"),
                    ],
                    "train",
                )
            if workflow == "train":
                run_training_wizard(teaching, cli.execute)
            else:
                run_validation_wizard(teaching, cli.execute)
            return
        except BackCommandError:
            print("已取消当前工作流，返回向导入口。")
            if cli.workflow:
                return
        except ExitCommandError:
            print("已退出，未执行后续操作。")
            return
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"操作失败：{exc}")
            return


if __name__ == "__main__":
    main()
