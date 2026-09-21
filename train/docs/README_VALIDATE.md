# MBAN 评估说明

本文件只负责软件评估、PTQ/MC、硬件压力、mapping 和 PPA；训练见 [`README_TRAIN.md`](README_TRAIN.md)，研究任务见 [`MBAN.txt`](MBAN.txt)。

## 0. 推荐入口：eval.yaml

多模型、多场景和多协议统一写入 `eval.yaml`，然后只执行：

```powershell
python evaluate_mban.py
```

也可以显式指定配置文件：

```powershell
python evaluate_mban.py --eval path\eval.yaml
```

`models.model_type` 选择本次评估的唯一类型；`models.items` 是要一起评估的 checkpoint 列表。所有模型都使用同一个类型，`display_name` 只用于结果展示，省略时自动使用 `model1`、`model2`……。`enabled.evaluations` 只选择本次运行的评估项，详细参数统一写在 `evaluations` 下。每项评估独立输出，`eval_manifest.json` 记录执行状态。

日常使用直接启动对话向导：

```powershell
python mban_wizard_cn.py --workflow validate
```

向导只询问模型、统一评估类型、评估项和活体帧数，生成 `results/eval/_plans/eval_selected.yaml` 后调用统一入口；高级参数仍在 `eval.yaml` 中维护。

```yaml
models:
  model_type: fp32
  items:
    - display_name: model1
      checkpoint: results/model_ablation/model/FP32/H32/OC16/best_val.pth

enabled:
  evaluations: [scenes, invivo]
```

上面配置运行一个 FP32 模型的四场景和固定 test split。当前投稿主线从 50 个 test 条目中确定性选取 10 帧报告活体指标；QAT、PTQ 需要分别建立对应的评估计划；同一计划中可以继续添加多个同类 checkpoint。

## 1. 统一口径

硬件配置来源按评估类型区分：`fp32` 不启用硬件；`qat` 使用 checkpoint 的
`hardware_qat_state.config`；`ptq` 和部署评估使用 `train/hardware_eval.yaml` 作为目标硬件配置。
因此 QAT checkpoint 若训练时通过 `--set bias_bits=32`，评估仍使用 32-bit；不需要修改评估 YAML。

```text
inter_layer=analog       # PTQ/mapping 默认；QAT 以 checkpoint 合同为准
bias_implementation=ordinary  # ordinary Bias 实际跟随 inter_layer
hardware_eval.yaml.hardware 中统一配置 `weight_bits/input_bits/control_bits/inter_layer_bits/bias_bits`
ordinary bias follows `bias_bits`; array bias follows `weight_bits`
per_output_channel=false
interpolation_bits=32
profiles={ideal, common}
stress.mc_runs=30
```

`display_name` 是本次评估的自定义展示名，不代表 checkpoint 内部身份；H32×OC8、H64×OC8 或其他候选必须使用各自 checkpoint 的配置和 manifest。当前 `results/main` 的 v31/v34 混合状态和投稿前重跑要求见 [`TSON_submission.md`](TSON_submission.md)。

`.pth` 是训练 checkpoint：`model_state_dict` 保存网络参数，`model_config/evaluation_config/geometry_config` 保存模型语义，QAT checkpoint 另存 `hardware_qat_state`。`checkpoint_identity.training_mode` 只标记 FP32 或 QAT 训练来源；PTQ 是评估后端，不是 checkpoint 身份。

`ideal` 只关闭额外硬件非理想；`common` 是当前 YAML 定义的代表性行为级压力，不是具体器件实测 worst-case。`common` 当前启用 IR-drop、4-bit control ADC noise/nonlinearity、drift、stuck-at、write/read noise、TIA 通道增益失配/TIA noise，以及 L∞ reference noise/静态增益失配。当前未启用的字段不能写成已验证结果。QAT 评估使用 checkpoint 保存的完整硬件合同；PTQ 才按 YAML 的目标位宽重新校准。

行为级 `fp32/qat/ptq` 评估不要求 checkpoint 已 fold；只有 deployment、严格硬件 MC、mapping、CrossSim 和 PPA 要求部署态 checkpoint。

## 2. 软件和量化评估

```powershell
cd E:\paper1\deeplearn\train

# 场景图像和指标
python evaluate_mban.py --mode scenes --model-type fp32 --model Candidate=path\best_val.pth --output results\candidate_scenes

# 固定 test split；完整计划统一在 eval.yaml 中指定 model_type
python evaluate_mban.py --mode test --model-type fp32 --model Candidate=path\best_val.pth `
  --split-file path\mixed_split.csv --test-frames 50 --output results\candidate_test

# PTQ4：--mode ptq 将模型列表统一作为 PTQ，校准只使用 train split
python evaluate_mban.py --mode ptq --model-type ptq --model Candidate=path\best_val_deploy.pth `
  --split-file path\mixed_split.csv --output results\candidate_ptq4

# 在同一后端的评估计划中加入 figures
python evaluate_mban.py --eval eval.yaml
```

固定 test 评估使用同一 split、ROI、display range 和 reference。活体主要报告逐帧 SSIM 的 mean/SD/CI；受控 contrast 场景报告 CR/CNR/gCNR，resolution 场景报告 lateral/axial FWHM 和 PSLR。不同 scene 不合并成一个均值。

固定 test 图表是报告输出模式，模型列表显式指定，DAS 作为统一基线：

```powershell
python evaluate_mban.py --mode figures `
  --model-type fp32 `
  --model FP32=path\\fp32\\best_val.pth `
  --split-file path\\mixed_split.csv `
  --test-frames 5 --output results\\figures
```

每个 test frame 使用同一套 `load_scene -> reconstruct` 生成模型图和 SSIM，另生成单角度 DAS 图；输出为 `metrics.csv`、`figure_manifest.json` 和逐帧 PNG。FP32、PTQ、QAT 都可以作为模型，不再绑定固定 H64/OC 配对。

## 3. 硬件压力、mapping 和 PPA

```powershell
# 参考量程校准：只使用 train calibration subset
python evaluate_mban.py --mode calibrate --model-type ptq --eval-config hardware_eval.yaml `
  --model Candidate=path\best_val_deploy.pth `
  --split-file path\mixed_split.csv --output results\hardware_calibration

# common：PTQ4 单因素与 30-run MC
python evaluate_mban.py --mode hardware --model-type ptq --config config.yaml `
  --eval-config hardware_eval.yaml --profile common `
  --model Candidate=path\best_val_deploy.pth `
  --model Direct=path\direct_best_val_deploy.pth `
  --split-file path\mixed_split.csv `
  --references-json results\hardware_calibration\nonideal_references.json `
  --mc-runs 30 --output results\common

# 定制紧凑 mapping（64×64 子阵 + 同输入层纵向打包）
python evaluate_mban.py --mode mapping --model-type ptq --model Candidate=path\best_val_deploy.pth `
  --output results\mapping --mapping-mode custom_compact --tile-rows 64 --tile-cols 64 `
  --converter-schedule parallel --differential-readout post_tia_subtractor

# PPA ledger；当前 YAML PPA 网格是 4-bit ADC
python evaluate_mban.py --mode ppa --model-type ptq --model Candidate=path\best_val_deploy.pth `
  --output results\ppa --mapping-mode custom_compact --tile-rows 64 --tile-cols 64 `
  --converter-schedule parallel --differential-readout post_tia_subtractor `
  --input-bits 4 --input-timing analog_single_pulse --adc-bits 4 --technology-node 32
```

MC 中 write/drift/stuck 状态在 realization 内固定，read/TIA/ADC 动态噪声按调用重采样；候选模型必须共享同一组 realization seeds。PTQ4、QAT 和硬件 MC 不能混写成同一种结果。

硬件链路 trace 只接受 `qat` 或 `ptq`：

```powershell
python evaluate_mban.py --mode trace `
  --model-type qat --model QAT=path\\qat\\best_val.pth `
  --frames 6 `
  --output results\\trace_qat

python evaluate_mban.py --mode trace `
  --model-type ptq --model PTQ=path\\fp32\\best_val.pth `
  --split-file path\\mixed_split.csv --frames 6 `
  --output results\\trace_ptq
```

`--model DISPLAY_NAME=PATH` 是临时单模型写法，必须同时指定 `--model-type`；完整实验使用 `eval.yaml` 的 `models.model_type`、`models.items` 和 `enabled.evaluations`。命令行临时多模型也必须使用同一种 model_type。PTQ 不写入 `.pth`，它是评估时对 FP32 checkpoint 施加硬件量化并用 train split 校准。

QAT 直接读取 checkpoint 的硬件合同；PTQ 在 trace 前只用 train split 校准量程。FP32 没有硬件后端，不能生成该 trace。`trace.json` 的控制路径按 `raw_control -> adc_control -> physical_weight` 记录，另含 ADC/TIA/偏置/权重映射诊断。

## 3.1 QAT 参数审计

训练后先审计 Bias 路径与普通权重的实际映射量程，不把 raw Bias 大小直接当成有效硬件 Bias：

```powershell
python tools/qat_audit.py `
  --model ordinary=path\ordinary\best_val.pth `
  --model array=path\array\best_val.pth `
  --output results\qat_audit
```

输出 `qat_parameter_audit.csv/json`，记录冻结 Bias 范围、有效 Bias、共享/核心量程、实际权重 LSB 和核心量化误差。日志序列使用：

```powershell
python tools/training_log.py --log path\train.log --output results\qat_audit\logs
```

## 4. 模拟、数字和公共边界

`inter_layer` 是全局互斥开关：

```text
Analog: TIA → dual/配置的模拟处理 → analog buffer → next FC
Digital: TIA → ADC → 数字 Bias/处理 → DAC/selector → next FC；最终层为 TIA → 输出 ADC → 数字 Bias
```

当前 YAML 默认 Analog；Digital 只能作为明确的候选配置重新评估，不能按 fc1/fc2 混合。`bias_implementation` 的请求值为 `ordinary` 或 `array`：`ordinary` 跟随 `inter_layer` 解析为 `none/analog/digital`；`none` 不建模层间 Bias 非理想，但 Bias 仍可能按 `bias_bits` 量化，`analog` 在 TIA 前走模拟 Bias，`digital` 在对应 ADC 后走数字 Bias。`array` 使用独立的常量输入列，并按 `weight_bits` 量化。主线 mapping 已分别生成 analog/digital ledger，但不代表物理电路已经验证。

增量自适应模块从“完成 TOF 对齐和 RMS 归一化的复数 IQ”开始，到 control readout、K→M interpolation 和 complex beamforming 为止。RF ADC、DDC、TOF、输入 RMS、envelope、TGC、log 和显示是公共链路：图像实验中保持一致，不计入自适应模块自身 PPA；完整系统指标必须另加公共链路。

## 5. PPA 解释

报告时必须分开：

- logical coefficients；
- G+/G− physical devices；
- 定制矩形子阵、同输入层纵向时间复用和实际 padding；
- conversion dimension；
- physical DAC/ADC count；
- CIM core、standard peripherals、custom add-on 和 full-flow estimate。

`inference_seconds` 是 Python 软件时间，不是芯片 latency。CrossSim 只作 crossbar-level 功能趋势验证，不承担完整网络 PPA；NeuroSim 结果不能自动包含 K→M、IQ weighting、公共前后端和未建模外围。
