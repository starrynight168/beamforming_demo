# Beamforming Demo

`beamforming_demo` 是一个基于 Python/PyTorch 实现的超声波束合成（Beamforming）对比与评估工程。本工程实现了多种主流波束合成算法，支持 GPU 加速，并对四个 controlled phantom 场景提供与 PICMUS (Platform for Interdisciplinary Research in Medical Ultrasound) 规则语义对齐的定量评估；in-vivo 场景用于成像对比，不执行官方靶标定量指标。

---

## 核心算法列表

本工程目前支持并实现了以下波束合成算法：
- **DAS (Delay and Sum)**: 延迟相加法，最基础、高效的经典波束合成方法。支持离散通道孔径与连续几何孔径模式。
- **MV (Minimum Variance)**: 最小方差自适应波束合成，基于数据自适应计算空间加权矢量，显著提升图像分辨率与对比度。
- **ESBMV (Eigenspace-Based Minimum Variance)**: 特征空间最小方差法，通过对协方差矩阵进行特征分解并投影到信号子空间，增强 MV 的稳健性。
- **GCF-MV (Generalized Coherence Factor - Minimum Variance)**: 广义相干因子自适应波束合成，利用 GCF 压制旁瓣和噪声。
- **F-DMAS (Filtered Delay Multiply and Sum)**: 滤波延迟相乘相加法，通过孔径内对信号进行两两相乘处理，有效降低主瓣宽度并抑制旁瓣。
- **CMSAW (Coherence-based MV variant)**: 基于自适应相干权重的最小方差流式优化方法，在大角度复合 CPI 下兼顾画质与显存占用。

---

## 项目目录结构

```text
beamforming_demo/
  ├── config.yaml               # 统一全局实验与算法默认超参数配置文件
  ├── run_one.py                # 单场景多算法批量重建、对比拼接与指标评估入口
  ├── run_all.py                # 一键运行配置中所有场景的批处理脚本
  ├── tools/                    # 辅助工具与专项实验脚本
  │    ├── run_wizard_cn.py     # 面向交互式超声成像配置的中文引导向导
  │    ├── run_ablation_cn.py   # 用于算法超参数调优与指标自动分析的消融实验工具
  │    └── ablate_sound_speed.py # 声速专项消融实验
  │
  ├── data/                     # 数据管理目录
  │    ├── pack_data.py         # 原始 PICMUS 格式数据打包为工程自描述 H5 的脚本
  │    ├── check_data.py        # 针对 H5 数据集类型、形状、完整性、NaN/Inf 的体检工具
  │    ├── simulation.h5        # 打包后的仿真数据（暗斑/散斑与分辨率/畸变靶点）
  │    ├── experiments.h5       # 打包后的水槽实验数据（暗斑/散斑与分辨率/畸变靶点）
  │    └── in_vivo.h5           # 打包后的在体颈动脉数据（横切面与纵切面）
  │
  ├── algorithms/               # 核心重建算法及公共工具模块
  │    ├── common.py            # 公共参数、H5 数据、数值、路径和输出工具
  │    ├── das.py               # DAS 重建脚本
  │    ├── mv.py                # MV 重建脚本
  │    ├── esbmv.py             # ESBMV 重建脚本
  │    ├── gcfmv.py             # GCF-MV 重建脚本
  │    ├── cmsaw.py             # CMSAW 重建脚本
  │    ├── fdmas.py             # F-DMAS 重建脚本
  │    └── template_algorithm.py # 添加新算法时的标准脚手架模版
  │
  ├── evaluation/               # 成像指标计算与绘图模块
  │    ├── evaluate.py          # controlled phantom 场景的 PICMUS 语义定量指标核心
  │    └── plot_metrics.py      # 指标对比柱状图、横向波束剖面图（Profile）自动绘制工具
  │
  └── results/                  # 重建图像与评估报告输出目录（自动创建）
```

---

## 环境安装与配置

推荐使用 Conda 建立独立的 Python 环境：

```bash
# 创建并激活 Conda 环境
conda env create -f environment.yml
conda activate beamforming-demo
```

也可以直接通过 pip 安装依赖项：

```bash
pip install -r requirements.txt
```

> [!NOTE]
> 本项目核心计算完全基于 PyTorch 张量运算。默认配置为 CUDA GPU 加速；若无 NVIDIA GPU，程序将自动回退到 CPU 执行计算。

---

## 数据集准备与体检

为了进行完整的算法比对，需要将 PICMUS 挑战赛数据导入本工程：

1. **下载原始数据**：
   从以下链接下载预整理好的 `PICMUS` 原始数据压缩包：
   - 链接：[PICMUS 数据压缩包](https://drive.google.com/file/d/1CQxjvpwGHDyzwHSJQux-mkXLl97mul-f/view?usp=drive_link)
   
2. **放置路径**：
   在项目根目录下创建并解压到 `data/` 目录，确保其目录布局为：
   `data/PICMUS/database/` 等。

3. **打包生成 H5 数据集**：
   运行打包工具，该脚本会执行 complex RMS 归一化，嵌入自描述元数据 `config_yaml`，并以 `float16` 存储 IQ。源数据必须提供有限正数的采样频率、声速、载频（`fc`/`modulation_frequency`）以及阵元间距（`pitch`/均匀 `probe_geometry`）；缺失时直接报错，不使用推测默认值。`initial_time` 可以是标量或与发射角数量一致的向量：
   ```bash
   python data/pack_data.py
   python data/pack_data.py --only simulation
   # --only 也支持 experiments 或 in_vivo
   ```

4. **进行数据完整性体检**：
   验证必需字段、数据类型、形状、有限性、归一化尺度、路径元数据和 contract 一致性。`valid_time_samples` 必须是有效整数，其后的 IQ 必须保持零填充；`all_envdb_norm` 必须为有限的 `[N,1,H,W]` 且位于 `[0,1]`：
   ```bash
   python data/check_data.py
   python data/check_data.py data/simulation.h5
   ```

---

## 运行指南

### 1. 单独运行指定算法进行成像
可以直接调用算法脚本对特定 H5 里的某个样本进行重建：
```bash
python algorithms/das.py --h5_path data/simulation.h5 --h5_sample_idx 0 --output_dir results/simulation_contrast_speckle
```
此操作将在 `results/simulation_contrast_speckle/das/` 下生成重建的矩阵 `das.npy` 和 B-Mode 图像 `das.png`。

### 2. 单个场景的一键批量比对与评估 (`run_one.py`)
使用 `run_one.py` 可以自动调度 `config.yaml` 中配置的所有成像方法进行同一场景的波束合成，并在场景根目录下生成包含 Ground Truth 的拼接对比图 `comparison.png`，同时导出高分辨率独立的个人 B-Mode 图像目录 `individual_images/`：
```bash
python run_one.py --scene simulation_contrast_speckle
```
重建完成后，controlled phantom 场景会自动启动评估系统，在 `metrics/` 目录下生成：
- **`summary_metrics.csv`**: 所有算法在当前场景的全面对比指标表。
- **`contrast_roi_metrics.csv` / `resolution_target_metrics.csv`**: 单个 ROI 和点目标的明细指标。
- **`contrast_group_metrics.csv` / `resolution_group_metrics.csv`**: 按场景分组的汇总指标。
- **`standard_metrics.png` / `auxiliary_metrics.png`**: 主要和辅助指标的柱状对比图。
- **`cyst_profile.png` / `point_profile.png` / `roi_targets.png`**: 可用时生成的剖面图和 ROI/靶标示意图。
- **`picmus_challenge_summary.txt`**: PICMUS 风格分组指标的文本总结。

in-vivo 场景仍会生成算法输出、`comparison.png` 和 `individual_images/`，但会跳过 phantom ROI、点目标和 PICMUS 分组定量评估。

### 3. 一键重建并评估全部场景 (`run_all.py`)
一次性运行项目内的全部仿真、实验和在体（in-vivo）场景；其中只有四个 controlled phantom 场景会生成完整定量评估：
```bash
python run_all.py
```
若只想执行部分场景，可使用命令行过滤：
```bash
python run_all.py --only simulation_contrast_speckle,carotid_cross
```

---

## 交互向导与参数实验工具

为了更方便地进行研究，本工程提供了两个功能强大的中文交互式脚本：

### 1. 成像引导向导 (`run_wizard_cn.py`)
为初学者或临时调参设计的图形化命令行向导。支持一步步中文提问：
- 交互式选择数据集、样本帧、波束合成方法及参数；
- 提供详细的小白背景说明，解释诸如 F-Number、TGC、窗函数和插值法对画质的实质影响；
- 支持自适应参数提示，根据所选算法动态询问其特有超参数（例如调节 ESBMV 的特征门限、F-DMAS 的时间平滑窗等），输入非法时实时拦截并报错。

运行命令：
```bash
python tools/run_wizard_cn.py
```

### 2. 参数消融实验工具 (`run_ablation_cn.py`)
针对学术研究设计的“控制单变量超参数消融”工具。可以快速获取特定超参数变化对重建质量的演变曲线：
- 自由选择要消融的方法专属超参数（如 MV 的对角加载因子、GCF 的低频 bins 数量、发射角度数等）；
- 支持指定具体的离散取值列表（如 `1, 3, 11, 75`），或使用等差生成器（如 `1.2:0.1:2.0`）；
- 自动完成批量消融重建后，在消融根目录统一调度评估，把每个参数值视作“不同算法”绘制直观的横向演变柱状图、点目标波束剖面对比曲线等，并集中输出高品质高清成像对比单图 `individual_images/`。

运行命令：
```bash
python tools/run_ablation_cn.py
```

### 3. 声速专项消融 (`tools/ablate_sound_speed.py`)
用于评估不同声速对单角度 MV 成像结果的影响：
```bash
python tools/ablate_sound_speed.py
```

---

## 评估指标体系说明

本工程内置的评估计算核心（`evaluation/evaluate.py`）在四个 controlled phantom 场景中对齐 PICMUS 挑战赛官方 MATLAB 算法语义：
- **对比度 (Contrast)**: 基于官方同心环形 ROI 划定，并且方差计算使用样本方差（$N-1$ 自由度），保障与 MATLAB 的 `var()` 结果完全一致；
- **散斑拟合度 (Speckle Quality)**: 提取散斑区进行 5 倍下采样并执行 Kolmogorov-Smirnov 检验以拟合 Rayleigh 分布，评估 KS 统计量 $D$ 和 $p$ 值；
- **分辨率 (Resolution)**: 提取点目标 lateral 剖面并线性插值至 $10\times$ 密度，计算 $-6\text{ dB}$ 的半高全宽（FWHM）；
- **几何畸变 (Distortion)**: 基于官方给定的累加标签掩膜、轴向修正因子和 7 个指定靶点位置自动判定畸变是否达标。
- **辅助学术指标**: 额外提供广义对比度噪声比（gCNR）、对比度噪声比（CNR）、峰值旁瓣电平（PSLR）、积分旁瓣电平（ISLR）以及基于参考图的图像结构相似度（SSIM）、峰值信噪比（PSNR）与平均绝对误差（MAE）等分析。

*(注：in-vivo H5 中的 `all_envdb_norm` 是打包阶段生成的多角度 DAS 参考图，不等同于带物理靶标的 PICMUS GT。因此在体颈动脉场景会自动跳过 phantom ROI、点目标和 PICMUS 分组定量，只做图像重建与对比拼接。)*

---

## 新增算法扩展指南

本工程设计有高度的可扩展性。只需遵循以下步骤即可快速接入并测试自己的超声波束形成新算法：

1. **复制模版**：
   ```bash
   cp algorithms/template_algorithm.py algorithms/my_method.py
   ```
2. **设定唯一算法 ID**：
   打开新创建的脚本，确保顶部的 `METHOD_NAME = "my_method"`，保证脚本名、METHOD_NAME、以及后续在 `config.yaml` 中配置的键值三者完全一致。
3. **实现核心重建逻辑**：
   在 `my_method.py` 中实现 `beamform(data, args)` 函数。该函数的输入 `data` 已经通过公共 H5 载入器（`common.py`）完成了网格映射与通道延迟的映射。您只需要读取 `data["I"]` / `data["Q"]`，按需要计算加权值（如自适应权重矩阵），并返回最终二维的 B-Mode 对数包络图像矩阵（对齐 `[z_grid, x_grid]`，峰值归一化至 `0 dB`）。
4. **配置默认参数**：
   若新算法有专属的控制超参数，首先在 `my_method.py` 内部定义 argparse 参数（例如 `--my_param`），然后在 `config.yaml` 根目录的 `algorithm_params` 中加入默认值：
   ```yaml
   algorithm_params:
     my_method:
       my_param: 0.5
   ```
   并在 `config.yaml` 顶部的 `algorithms` 列表中追加 `"my_method"`。
5. **一键测试与多维评估**：
   参数配置完成后，新算法将被全局识别并可以与 DAS、MV 等算法同时跑对比：
   ```bash
   python run_one.py --scene simulation_contrast_speckle --algorithms das,mv,my_method
   ```
   重建系统和评估系统将完全自动生成对应的对比子图，并将其横向指标绘制到对比表和 Profile 曲线中，无需修改任何绘图或控制流代码。
