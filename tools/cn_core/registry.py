from .interactive import ROOT


TOOLS = (
    {
        "name": "RF/解析 RF/IQ/STFT 频带敏感性对比",
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
        "script": "ablate_sound_speed.py",
        "output_name": "ablation_sound_speed",
        "basic": (
            ("reference_key", "选择参考图", "reference", None),
            ("c_min", "搜索声速下限（m/s）", "positive", 1430),
            ("c_max", "搜索声速上限（m/s）", "sound_max", 1620),
        ),
        "advanced": (
            ("sections", "区间分段数（至少 3）", "sections", 4),
            ("tol", "结束区间宽度（m/s）", "positive", 2),
            ("max_iter", "最大迭代次数", "positive_int", 6),
            ("grid_step", "附加网格步长（0 关闭）", "nonnegative", 15),
            ("mv_dl", "对角加载系数", "nonnegative", 0),
            ("subarray_ratio", "子阵长度比例", "ratio", 0.25),
            ("temporal_win", "时间窗口（正奇数）", "odd", 9),
            ("f_number", "动态孔径 F-number", "positive", 1.5),
            ("no_dynamic_aperture", "接收孔径", "flag", ((False, "动态孔径"), (True, "全孔径"))),
            ("no_fbss", "前后向空间平滑", "flag", ((False, "开启"), (True, "关闭"))),
            ("no_tgc", "TGC 深度补偿", "flag", ((False, "开启"), (True, "关闭"))),
            ("interp", "插值方式", "choice", (("linear", "线性"), ("cubic", "三次"))),
        ),
    },
)

