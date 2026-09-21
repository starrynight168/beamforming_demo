# 单角度超声 IQ 学习与自适应波束形成

本项目围绕超声平面波成像中的单角度输入学习任务展开：以单角度 IQ 信号作为网络输入，学习高质量自适应波束形成结果，并保留多角度 DAS 等传统重建结果作为参考。

项目同时包含数据打包、传统/自适应波束形成算法、训练脚本、结果评估与 H5 数据检查工具，方便在仿真、PICMUS、EPFL 体内/体外数据上进行统一实验。

## 项目结构

```text
deeplearn/
├─ README.md                # 项目总览与统一工作流
├─ config.yaml              # 根目录算法对比、场景和显示参数
├─ run_one.py               # 单场景算法运行与评估
├─ run_all.py               # 多场景批量运行
├─ tools/                   # 辅助工具
│  ├─ run_ablation_cn.py    # 算法/场景消融批处理
│  └─ run_wizard_cn.py      # 算法运行向导
├─ algorithms/              # DAS、MV、ESBMV、GCF-MV、CMSAW、F-DMAS、MBAN 及公共工具
├─ data/
│  ├─ scripts/              # EPFL/PICMUS 数据打包与仿真脚本
│  ├─ checks/               # H5 字段、尺度、数据完整性检查工具
│  ├─ EPFL/                 # EPFL 原始 npz 数据目录
│  └─ *.h5                  # 打包后的训练/评估数据
├─ train/                   # MBAN 训练、软件评估和硬件感知评估
│  ├─ config.yaml           # 训练、数据、模型和软件运行模式
│  ├─ hardware_eval.yaml    # 硬件默认值、profile 和评估矩阵
│  ├─ evaluate_mban.py      # 软件/PTQ/MC/硬件统一评估入口
│  ├─ protocols/            # 特殊实验协议
│  ├─ benchmarks/           # 性能/数值 benchmark
│  ├─ eval_core/            # 统一评估、图表和硬件链路诊断
│  ├─ tests/                # MBAN 配置、硬件和 beamforming 回归测试
│  ├─ docs/                 # 训练、评估和论文实验口径
│  ├─ server_jobs/          # 云端训练与消融实验批处理入口
│  └─ results/              # MBAN 实验与消融资产
│     ├─ SI/                # 10 项核心微观/结构消融基线与评测 (README.md 详述)
│     └─ model_ablation/    # 3×7 骨干容量与控制自由度 Pareto 消融 (README.md 详述)
├─ evaluation/              # 指标计算与绘图
└─ requirements.txt         # Python依赖
```

配置文件职责必须分开：根目录 `config.yaml` 只控制传统算法和场景；`train/config.yaml`
控制 MBAN 训练与软件模型；`train/hardware_eval.yaml` 控制 QAT/PTQ 使用的硬件默认参数、
非理想 profile 和 mapping/CrossSim/PPA 评估矩阵。不要把三类配置混写。

## 环境与执行约定

在项目根目录执行命令。依赖可使用 `requirements.txt` 或 `environment.yml` 安装；GPU训练还需要与本机 CUDA/PyTorch 匹配的环境。所有路径示例均相对于 `E:\paper1\deeplearn`，也可以替换为绝对路径。

建议的最小顺序是：

```text
准备/打包 H5
→ check_data.py --full 生成并固定检查凭证
→ run_one.py / run_all.py 运行传统算法和场景评估
→ train/mban.py 训练 FP32 MBAN
→ train/evaluate_mban.py 做 test/PTQ/硬件评估
→ evaluation/ 生成统一指标表和图
```

训练和正式评估都应固定 `seed`、数据 split、checkpoint、配置快照和代码版本。不要直接覆盖已有结果目录。

辅助工具从根目录运行：

```text
python tools/run_wizard_cn.py
python tools/run_ablation_cn.py
```

## 数据打包目标

默认打包逻辑面向“单角度输入，学习高质量 teacher”的训练范式：

```text
输入：
  单角度 baseband IQ

主 teacher：
  MV

参考 teacher：
  多角度 DAS
```

打包配置位于：

```text
data/scripts/pack_config.yaml
```

核心配置示例：

```yaml
outputs:
  main: mv
  references:
    - das
  angle_select:
    mv: input
    das: all
  save_complex:
    - mv
  save_envdb:
    - mv
    - das
  save_amp: []

pack:
  input_angles: 1
  norm_mode: rms
  preview_root: ../gt_images
```

其中：

- `pack.input_angles` 控制保存到 H5 的输入角度数；
- `outputs.angle_select` 控制每个 teacher 使用的重建角度；
- `save_envdb` 保存显示域 B-mode 标签；
- `save_complex` 保存复数 IQ teacher；
- `save_amp` 可选保存线性幅值。

运行 EPFL 打包：

```powershell
python data\scripts\pack_EPFL.py --config data\scripts\pack_config.yaml
```

打包脚本支持单个 `npz`、多个 `npz`、文件夹和多个 dataset 配置。已有 H5 会增量续写，并在恢复时自动检测和压缩断点空槽。

EPFL 的 HDF5 压缩由 `pack_config.yaml` 的 `compression` 控制，可选 `none`、`gzip`、`lzf`、`blosc_zstd`；PICMUS 仿真的压缩参数直接在 `simulate_PICMUS.py` 文件顶部设置。

## H5 主要字段

当前 H5 字段以训练和复现实验为核心：

```text
all_multi_I              # 输入 IQ 实部，shape=[N,A,T,C]
all_multi_Q              # 输入 IQ 虚部，shape=[N,A,T,C]
time_start_vector        # 每个输入角度的 t0，shape=[N,A]
valid_time_samples       # 每个样本的有效时间长度，shape=[N]；其后 IQ 必须为零填充
angles                   # H5 保存的输入角度

all_envdb_norm           # 主 teacher 显示域标签，默认 MV
all_envdb_das_norm       # DAS 参考显示域标签

mv_i
mv_q                     # MV 复数 IQ teacher

all_scale_ref            # 输入 IQ 归一化尺度
all_norm_ref             # 主 teacher 相对输入尺度

acquisition_id           # 样本 ID，例如 invivo_15002
body_region              # 部位，例如 carotid
input_angle_source       # 输入角度策略
gt_angle_source_by_alg   # 各 teacher 的角度策略
n_written                # 有效写入样本数
```

这里的`*_ref`是幅度归一化/恢复用的尺度因子，不是另一张GT图，也不是评估指标；`norm_references`则是H5元数据中“算法名→尺度字段”的映射。

GT口径：四个受控场景的完整指标使用`all_envdb_norm`；EPFL打包数据默认该字段为MV显示域GT，训练监督对应`config_yaml.generation.training_targets.complex_iq_targets`中的`mv_i/mv_q`。PICMUS仿真只生成DAS，因此其`all_envdb_norm`为DAS。`all_envdb_das_norm`作为DAS参考保留。

### 两级 IQ 尺度处理

`pack_EPFL.py` 在离线打包时为每个输入记录计算全局 `scale_ref`，并用同一尺度归一化输入 IQ 与监督目标，保存到 `all_scale_ref`。这是数据集尺度标定，不是推理时的硬件 RMS 模块。

训练/推理阶段的 `train/config.yaml:input.input_normalization` 默认使用 `rms`，按当前像素和有效动态孔径重新计算局部 RMS；波束合成后恢复 `input_scale`。实际硬件只需要实现一次局部 RMS，或用固定增益/AGC替代。若修改为 `none` 或 `std`，必须重新训练并在结果中记录该输入尺度协议。

## 算法

`algorithms/` 中包含：

```text
DAS
MV
ESBMV
GCF-MV
CMSAW
F-DMAS
```

算法对比配置位于：

```text
config.yaml
```

运行单个场景：

```powershell
python run_one.py --scene invivo_15002
```

只运行指定算法：

```powershell
python run_one.py --scene invivo_15002 --algorithms das,mv,cmsaw --no_evaluate
```

默认结果保存到：

```text
results/<scene_id>/
```

批量运行根目录 `config.yaml` 中的场景：

```powershell
python run_all.py --only all
python run_all.py --only simulation_contrast_speckle,simulation_resolution_distorsion --algorithms das,mv,cmsaw
```

### 动态孔径的统一对照与 DAS 专用模式

默认算法对比使用按通道数扩张的离散动态孔径（`discrete`）：DAS 与 F-DMAS 均采用和 MV、ESBMV、GCF-MV 相同的离散有效通道数。这样可以排除孔径筛选规则不同造成的影响，使分辨率、旁瓣和散斑的差异主要反映算法本身；也避免 MV 在协方差估计和求解之外增加连续几何孔径的额外计算。

几何孔径不作为公共模式。DAS 保留下列专属选项，用于单独研究孔径裁剪规则：

- `discrete`：默认值，与其他算法的统一对照设置一致。
- `geometry`：连续几何半宽裁剪，仅用于考察几何掩膜孔径本身的影响；不建议直接与默认 MV 作严格公平比较。

在 `config.yaml` 中设置：

```yaml
algorithm_params:
  das:
    aperture_mode: discrete  # 或 geometry
```

临时使用几何模式：

```powershell
python run_one.py --scene invivo_15002 --algorithms das --aperture_mode geometry
```

`data/scripts/pack_EPFL.py` 中用于生成 DAS teacher 的内部实现固定采用离散通道孔径，并且会在延时采样越界后对有效窗权重重新归一化。因此当前打包出的 DAS 标签与默认对照设置一致；无需为打包流程增加 `geometry` 模式。

## CMSAW 复数 IQ 说明

CMSAW 本身主要是基于 MV baseline 的幅值加权方法。项目中提供了统一的 `CMSAWBeamformerIQ` 接口，使其可以作为 pack-time teacher 使用：

```text
S_cmsaw = S_mv × weight_cmsaw
```

因此：

```text
CMSAW 的 envelope / dB 图来自 CMSAW 幅值加权；
CMSAW 的 phase 继承 MV。
```

这适合用于 B-mode / envdb 训练目标。如果实验关注真实复相位、多普勒或相干相位分析，应明确该 IQ 是“MV 相位 + CMSAW 幅值”的复数表示。

## 训练

主要训练入口：

```text
train/mban.py
```

示例：

```powershell
python train\mban.py `
  --config train\config.yaml `
  --mode software

# BR/动态归一化 checkpoint 先折叠为唯一硬件部署格式
python train\evaluate_mban.py --mode checkpoint fold path\best_val.pth path\best_val_deploy.pth

# PTQ 不训练，使用统一评估入口校准并测试
python train\evaluate_mban.py --mode ptq --config train\config.yaml --eval-config train\hardware_eval.yaml --model MBAN=path\best_val_deploy.pth --split-file path\mixed_split.csv --output results\mban_ptq4 --set weight_bits=4 --set input_bits=4 --set control_bits=4

# QAT/HWA-QAT：必须记录训练配置、硬件评估配置和实际 profile
python train\mban.py --config train\config.yaml --eval-config train\hardware_eval.yaml --mode qat --profile common --set resume=none --set initial_checkpoint=path\fp32\best_val_deploy.pth
```

训练脚本会在每个 H5 内按固定随机种子划分训练、验证和测试集合，并保存 split CSV，保证后续复现实验使用相同划分。选中样本在数据集初始化时一次性载入 CPU 内存，后续 epoch 不重复读取 H5；GPU 几何缓存按 `(time_start_vector, valid_time_samples)` 动态去重，有多少种唯一几何就建立多少份缓存，相同几何的样本共享一份。显存不足的几何单独回退为实时计算。

正式结果必须同时保存：实际命令、checkpoint、split CSV、`effective_config.json`、训练日志和硬件评估时的 `hardware_eval_config_used.json`。`train/config.yaml` 当前仓库默认值为 `scheduler=onecycle`、`optimization_batch_pixels=16384`、`resume=all`；正式论文重跑若使用其他 override，必须以实际运行记录为准，不能混用不同配置产生的 checkpoint 和图表。

注意：`resume=all` 会在输出目录存在 `latest.pth` 时恢复训练状态。需要从冻结的 FP32 checkpoint 开始做新实验时，显式使用新的输出目录、`--set resume=none`，并按文档指定 `initial_checkpoint`。

## 数据检查

检查工具位于：

```text
data/checks/
```

常用命令：

```powershell
python data\checks\check_data.py --no_log
python data\checks\check_data.py --full --chunk-mb 128 data\volunteer_005.h5
python data\checks\check_single_angle_scale.py --no_log
```

这些脚本默认检查 `data/` 下的 H5；也可以显式传入 H5 路径。`--full`会分块检查全部RF/GT并生成训练所需的`.h5.check.json`凭证。

## 仿真与 PICMUS

PICMUS 仿真脚本：

```text
data/scripts/simulate_PICMUS.py
```

仿真与 EPFL 打包共用同一套核心重建参数，包括：

```text
input_angles
norm_mode
f_number
dynamic_aperture
tgc
tgc_alpha
window
interp
outputs.angle_select
```

这样可以让仿真数据、体外数据和体内数据在训练及算法对比时保持一致的重建设定。

## 评估与可视化

`run_one.py` 会生成算法结果、对比图和参数记录。开启评估时会调用：

```text
evaluation/evaluate.py
evaluation/plot_metrics.py
```

对于包含 GT 和标注的场景，可以计算参考指标和 ROI/target 指标；对于体内数据，主要用于定性对比和展示。

MBAN 评估统一使用 `train/evaluate_mban.py`：

```powershell
# 受控场景/活体软件评估
python train\evaluate_mban.py --mode scenes --config train\config.yaml --model MBAN16=path\best_val.pth --output results\mban_scenes

# PTQ4（硬件入口只接受 folded checkpoint）
python train\evaluate_mban.py --mode ptq --config train\config.yaml --eval-config train\hardware_eval.yaml --model MBAN16=path\best_val_deploy.pth --split-file path\mixed_split.csv --output results\mban_ptq4 --set weight_bits=4 --set input_bits=4 --set control_bits=4

# 硬件行为级 stress；需要先完成 calibrate 并提供 references JSON
python train\evaluate_mban.py --mode calibrate --config train\config.yaml --eval-config train\hardware_eval.yaml --model MBAN16=path\best_val_deploy.pth --split-file path\mixed_split.csv --output results\hardware_calibration
python train\evaluate_mban.py --mode hardware --config train\config.yaml --eval-config train\hardware_eval.yaml --model MBAN16=path\best_val_deploy.pth --model Direct160=path\direct160\best_val_deploy.pth --split-file path\mixed_split.csv --references-json results\hardware_calibration\nonideal_references.json --mc-runs 30 --output results\hardware_stress
```

`hardware_eval.yaml` 当前是 system-level behavioral hardware-aware evaluation：包含 PTQ4、IR-drop、ADC、normalized drift stress、stuck-at、paired Monte Carlo，以及可选的 write/read noise 补充项；不等同于真实忆阻器、TCAD、SPICE 或流片结果。未定义的 realistic profile、器件工艺参数和外围电路参数不能在论文中宣称为已完成硬件仿真。

训练、软件评估和硬件评估的详细口径分别见：

```text
train/docs/README_TRAIN.md
train/docs/README_VALIDATE.md
train/docs/PROJECT_MAP.md
train/docs/MBAN.txt
```

## 推荐实验范式与消融资产索引

基础训练设置：

```text
输入：单角度 IQ
主目标：MV
参考：多角度 DAS
```

扩展实验：

```text
ESBMV / GCF-MV / CMSAW / F-DMAS 作为额外 teacher 或算法对比
```

默认不建议一次性把所有 teacher 都写入正式训练包。更稳妥的做法是先保存核心字段，再根据实验目标增加额外 teacher。

### 1. 结构与微观消融体系 (SI Ablations)
项目在 `train/results/SI/` 下系统完成了 10 大核心模块的闭环消融实验（涵盖模型检查点与 4 大物理场景评测报告）：
- **activation**: $3 \times 2$ 矩阵（ReLU, PWL, Tanh $\times$ Single, Dual 支路），证实 Dual 架构在所有激活函数下均显著超越 Single。
- **envelope**: 线性包络监督损失权重扫描（$w_{\text{env}} \in [0.0, 1.0]$，步长 0.1），确定 $w_{\text{env}}=0.1$ 为最优平衡拐点。
- **loss**: 组合损失项消融（MSE, L1, Charbonnier 与平滑度/范围正则约束）。
- **input_norm**: 信号输入归一化方案（RMS、Std 与 None 对比，RMS 在动态孔径与物理增益下表现最优）。
- **bias**: 随层间路径切换的 Bias 配置（None、Analog、Digital，另有 Array Bias）。
- **normalization**: 隐藏层归一化行为（None, L1, L2, BatchRenorm 等在低比特硬件量化下的鲁棒性）。
- **interpolation**: 控制点到物理孔径的插值策略（Linear, Nearest, Cubic 对比）。
- **outlimit**: 输出值域约束（Nonnegative vs Signed, Hard vs Free 约束）。
- **qattext**: QAT 训练演进轨迹与 Schedule 阶段消融（Ideal 预热 + Common 硬件感知优化）。
- **windows**: 孔径窗函数与权重平滑窗口对比。

详细消融数据表格、物理机制剖析及各场景指标参见：[`train/results/SI/README.md`](train/results/SI/README.md)。

### 2. 模型容量与控制点 Pareto 消融 (Model Ablation)
项目在 `train/results/model_ablation/` 下构建了 $3 \times 7 = 21$ 个模型的超算网格：
- **Hidden Width ($H$)**: 32, 48, 64
- **Output Controls ($K$)**: 8, 16, 24, 32, 48, 64, 160 (Direct-Full)
- **核心结论**: $K=8$ 处存在极为明显的“收益悬崖”与 Pareto 拐点，仅凭 8 个控制点即可达到全通道 160 维输出 99.4% 的成像保真度，同时显著削减硬件 crossbar 阵列与 ADC 读出开销。

详细 Pareto 曲线、FP32/QAT 对比及完整指标数据参见：[`train/results/model_ablation/README.md`](train/results/model_ablation/README.md)。
