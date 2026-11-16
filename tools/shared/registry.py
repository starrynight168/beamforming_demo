from .interactive import ROOT


TOOLS = (
    {
        "name": "四场景复数输入归一化对比",
        "entry": "tools.code.compare_input_normalization",
        "script": "compare_input_normalization.py",
        "mode": "native",
    },
    {
        "name": "点靶重建图与分辨率指标",
        "entry": "tools.code.generate_point_target_images",
        "script": "generate_point_target_images.py",
        "mode": "native",
    },
    {
        "name": "RF/解析 RF/IQ/STFT 频带敏感性对比",
        "entry": "tools.cn_core.compare_rf_iq_cn",
        "script": "compare_rf_iq.py",
        "output_name": "rf_iq_comparison",
        "basic": (
            ("rf_source", "原始 RF 的 NPZ 文件或数据集文件夹", "rf_source", None),
            ("settings_dir", "原始 RF 的 settings 文件夹", "settings", ROOT / "data/EPFL/settings"),
            ("display_scale", "图像归一化方式", "choice", (("independent", "标准 B-mode（每幅按本幅峰值归一化）"), ("shared", "以 RF 峰值为参考（幅度对比）"))),
        ),
        "advanced": (
            ("mv_dl", "对角加载系数", "positive", 0.01),
            ("subarray_ratio", "子阵长度比例", "ratio", 0.25),
            ("temporal_win", "时间窗口（正奇数）", "odd", 3),
            ("f_number", "动态孔径 F-number", "positive", 1.5),
            ("no_dynamic_aperture", "接收孔径", "flag", ((False, "动态孔径"), (True, "全孔径"))),
            ("no_fbss", "前后向空间平滑", "flag", ((False, "开启"), (True, "关闭"))),
            ("interp", "插值方式", "choice", (("cubic", "三次"), ("linear", "线性"))),
            ("dr", "显示动态范围（dB）", "positive", 60),
            ("iq_gain", "共用尺度时 IQ 幅度校正倍数", "positive", 2),
            ("stft_window", "STFT 窗口采样点（偶数）", "positive_int", 16),
        ),
    },
    {
        "name": "声速搜索与消融",
        "entry": "tools.code.ablate_sound_speed",
        "script": "ablate_sound_speed.py",
        "mode": "native",
    },
    {
        "name": "重建网格扫描",
        "entry": "tools.code.rebeamform_invivo_grid",
        "script": "rebeamform_invivo_grid.py",
        "mode": "native",
    },
    {
        "name": "MV 对角加载点靶消融",
        "entry": "tools.cn_core.evaluate_mv_dl_sweep_cn",
        "script": "evaluate_mv_dl_sweep.py",
        "output_name": "mv_dl_sweep",
        "h5_argument": "data",
        "sample_argument": "sample-index",
        "output_argument": "output-dir",
        "direct_output": True,
        "basic": (
            ("meta", "点靶评价元数据 JSON", "file", None),
        ),
        "advanced": (
            ("grid-factor", "横向网格加密倍数", "positive_int", 8),
            ("window-mm", "点靶附近分析窗口（mm）", "positive", 1.8),
            ("subarray-ratio", "子阵比例", "ratio", 0.25),
            ("temporal-win", "时间协方差窗口（正奇数）", "odd", 9),
            ("dls", "对角加载候选值（逗号分隔）", "text", "0,0.0001,0.0003,0.001,0.003,0.01,0.03"),
            ("device", "计算设备", "choice", (("auto", "自动"), ("cpu", "CPU"), ("cuda", "CUDA"))),
        ),
    },
    {
        "name": "算法运行向导",
        "entry": "tools.code.run_wizard",
        "script": "run_wizard.py",
        "mode": "native",
    },
    {
        "name": "算法与场景消融",
        "entry": "tools.code.run_ablation",
        "script": "run_ablation.py",
        "mode": "native",
    },
    {
        "name": "归一化示意图",
        "entry": "tools.cn_core.normalization_demo_cn",
        "script": "normalization_demo.py",
    },
)

