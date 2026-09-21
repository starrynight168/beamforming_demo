# MBAN 工程索引

## 当前代码口径

以 `config.yaml`、`hardware_eval.yaml` 和 checkpoint 元数据为准；投稿结果审计见 [`TSON_submission.md`](TSON_submission.md)。论文草稿与历史结果中的旧版本只作记录。

| 项目 | 当前值 |
|---|---|
| 当前代码语义版本 | `MODEL_SEMANTICS_VERSION=40` |
| 结果资产语义版本 | `results/main` 同时存在 v31 与 v34；统一前不得作为最终投稿数值 |
| YAML 默认候选 | hidden width 32、hidden layers 2、dual、ReLU-hard、output controls 8 |
| 投稿候选 | `results/main` 当前为 H32×OC8 MBAN；Direct160/Project160 为对照，需统一语义版本后冻结 |
| 孔径 | dynamic aperture、`f_number=1.5`；network channels=160，mapping physical channels=192，线性 K→M interpolation |
| 输入 | packed complex IQ、RMS normalization |
| 输出实现 | E/explicit 为主线并生成物理权重；F/factorized 为部署路径，直接用 K 个 controls 合成，不保留 M 维权重 |
| 硬件压力 | `hardware_eval.yaml` 的 `common`、MC30；行为级压力，不是器件 worst-case |
| 主线活体统计 | 固定 test split 的 10 帧；split 总 test 条目为 50，不能混称 |

## 入口与依赖方向

```text
mban.py
  └─ mban_core/{config,data,model,beamforming,hardware,training}.py

evaluate_mban.py
  └─ eval_core/
     ├─ {config,inference,software,diagnostics,hardware,hardware_trace,hardware_ppa}.py
     └─ static_k8.py

tests/
  └─ 配置、硬件后端和 beamforming 回归测试

run_one.py
  ├─ algorithms/{das,mv,esbmv,gcfmv,cmsaw,fdmas}.py
  └─ evaluation/mban.py
```

- `mban.py` 是唯一训练入口；`evaluate_mban.py` 是唯一 MBAN 评估入口。
- 根目录 `run_one.py`/`run_all.py` 负责传统算法场景对比；不与训练入口合并。
- `evaluation/mban.py` 是根对比流水线适配器，复用 `eval_core` 的 MBAN 加载和 reconstruct，不维护第二套模型推理。
- `mban_wizard_cn.py` 是训练与统一评估的交互入口；评估向导只生成选择计划，不承载评估实现。
- `eval_core/` 承载统一评估、图表、硬件链路诊断，以及静态 K8 特殊协议。
- `tools/` 只承载可复现的 QAT 参数审计和训练日志提取，不重复实现模型推理。
- `evaluate_mban.py --mode figures`、`--mode trace` 是图表和硬件 trace 的统一入口。
- 单项调试可用 `--model DISPLAY_NAME=PATH --model-type TYPE` 临时指定；完整实验使用 `eval.yaml` 的统一 `models.model_type` 和 checkpoint 列表。
- 多任务实验使用默认 `eval.yaml` 或 `--eval path\eval.yaml`；`enabled.evaluations` 选择评估项，所有 checkpoint 在同一计划中使用同一种 model_type。
- `tests/` 使用 `python -m unittest discover -s train/tests -p 'test_*.py'` 执行。
- H5、split CSV、checkpoint、评估 CSV/JSON 和图像属于运行资产，不应提交到源代码目录。

## 文件处置规则

- 可删除生成缓存：`__pycache__/`、`.ruff_cache/`。
- `备份/` 是历史 checkpoint、结果和压缩包归档，不是当前代码依赖；在确认外部备份后再整体清理。
- 不删除 `data/` 原始 H5、`.h5.check.json`、当前配置、源代码和外部仿真依赖。
- MBAN 生成结果统一放在 `train/results/`，传统算法结果保留在根目录 `results/`；每次运行保留 `effective_config.json`、实际命令、checkpoint 语义版本和评估配置快照。

## 实验与消融资产结构 (`train/results/`)

### 1. 结构与微观机制消融 (`train/results/SI/`)
详见 [`train/results/SI/README.md`](../results/SI/README.md)，包含 10 项已完成的闭环消融：
- `activation/`: 3×2 激活矩阵 (`relu`, `pwl`, `tanh` × `single`, `dual`)
- `envelope/`: 线性包络损失权重扫描 ($w_{\text{env}} \in [0.0, 1.0]$)
- `loss/`: 组合损失函数消融 (MSE, L1, Charbonnier, 平滑正则)
- `input_norm/`: 输入归一化方案 (RMS, Std, None)
- `bias/`: 偏置实现（请求值 `ordinary/array`；`ordinary` 解析为 None/Analog/Digital）
- `normalization/`: 隐藏层归一化 (None, L1, L2, BatchRenorm)
- `interpolation/`: 控制点插值策略 (Linear, Nearest, Cubic)
- `outlimit/`: 输出值域约束 (Nonnegative vs Signed, Hard vs Free)
- `qattext/`: QAT 轨迹与 Schedule 阶段消融
- `windows/`: 孔径窗函数对比

### 2. 模型容量与控制点 Pareto 消融 (`train/results/model_ablation/`)
详见 [`train/results/model_ablation/README.md`](../results/model_ablation/README.md)：
- 覆盖 $H \in \{32, 48, 64\} \times K \in \{8, 16, 24, 32, 48, 64, 160\}$ 共 21 个模型网格。
- v31 资产覆盖 H32/H48/H64 × K=8/16/24/32/48/64/160；用于补充规模趋势，不能替代统一后的主线三候选。

### 3. 主线论文图包 (`train/results/main/figure/`)
- `figure1/plot_figure1_architecture.py`：根据当前部署口径绘制方法与系统边界；只读取 `train/results/main/evaluation/` 下的主线定制紧凑 mapping ledger。
- `figure2/plot_figure2_control_space.py`：读取主线 FP32 场景的控制量和物理权重曲线，绘制 K→M 机制图；DAS/MV 作为 M-only 物理基线，左图保留各模型原生 K 槽位，实际物理通道和动态孔径从记录数据读取。
- `figure2/plot_mainline_additional.py`：读取主线 FP32、PTQ ladder、QAT4、single-factor 和 mapping 结果，生成 Fig. 3–10。
- 主线评估数据位于 `train/results/main/evaluation/`，主线 checkpoint 位于 `train/results/main/model/`；QAT4 与 `ppa_mapping/` 已落盘，但 Fig. 3–9 的最终数值仍受主线 v31/v34 一致性审计约束。
- `train/results/model_ablation/` 与 `train/results/SI/` 属于补充/消融资产；其四场景和活体 CSV/PNG 不得冒充主线数据。
- 附加图工具和图像归档于 `train/results/main/figure/figure2/supplementary/`，并按 SI1、SI2 和消融模块分别保存。
- 主线图必须绑定实际存在的 PTQ、common 单因素、HWA-QAT、mapping/PPA 结果；缺失数据只能标记为未生成，不能补写数值。

## 复核重点

1. 训练和评估必须使用同一语义版本及同一 checkpoint 元数据。
2. PTQ calibration 只使用 train split；test split 只用于最终报告。
3. Analog/Digital 是全局层间路径，不能按 `fc1`、`fc2` 混用。
4. `best_val.pth` 是论文/部署主 checkpoint，`latest.pth` 只用于续训，`best_train.pth` 不作为主结果。
5. `CrossSim`、NeuroSim 和 PPA ledger 是硬件边界评估，不等于完整芯片延迟或端到端功耗。
6. 当前结果的 checkpoint 版本、配置差异和重跑要求见 [`TSON_submission.md`](TSON_submission.md)。
