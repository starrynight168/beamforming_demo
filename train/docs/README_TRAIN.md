# MBAN 训练说明

本文件只负责训练、QAT 和 checkpoint；研究协议见 [`MBAN.txt`](MBAN.txt)，评估命令见 [`README_VALIDATE.md`](README_VALIDATE.md)。

## 1. 配置依据

新实验只以以下 YAML 为准：

- `train/config.yaml`：模型、数据、损失和训练过程；
- `train/hardware_eval.yaml`：量化、层间路径、bias 实现、非理想 profile、QAT schedule 和 PPA 网格。

当前 YAML 默认值不是论文最终主线。投稿结果目录当前采用 H32×OC8 MBAN 作为候选、Direct160/Project160 作为对照；具体冻结状态和 v31/v34 混用问题见 [`TSON_submission.md`](TSON_submission.md)。

| 类别 | 当前 YAML 默认值 |
|---|---|
| 输入 | aligned complex IQ，`input_normalization=rms` |
| 骨干 | `hidden_width=32`，`hidden_layers=2`，`branch_mode=dual`，`dropout=0.2` |
| 隐藏层处理 | `centering=none`，`normalization=none`，`normalization_layers=[]`，`fc1/fc2=relu` |
| 输出 | 实值、`output_controls=8`、linear、nonnegative、hard unity |
| 控制归一化 | `control_normalization=none` |
| 孔径 | `dynamic_aperture=true`，`f_number=1.5` |
| 目标和损失 | MV complex IQ；IQ weight 1.0，envelope weight 0.1，其他默认关闭 |
| 硬件路径 | `inter_layer=analog`；`bias_implementation=ordinary` 时 Bias 跟随层间模式 |
| 硬件位宽 | `weight_bits/input_bits/control_bits/inter_layer_bits/bias_bits` 均由 `hardware_eval.yaml.hardware` 统一配置 |
| Bias | 请求值为 `ordinary` 或 `array`；`ordinary` 随 `inter_layer` 解析为 `none/analog/digital`，`array` 使用独立常量输入列 |
| 网络/插值 | `per_output_channel=false`，`interpolation_bits=32` |
| 硬件映射 | 默认使用 64×64 定制紧凑映射；该网格属于 mapping/PPA 配置，不是网络层尺寸 |
| QAT | 5 epochs `ideal` + 95 epochs `common` |

`hardware_eval.yaml` 的 `active_profile` 默认是 `ideal`；压力评估和 QAT 第二阶段使用 `common`。当前 profile 只有 `ideal` 和 `common`，不再使用旧的 `max_coverage` 名称。

已有 checkpoint 必须以自身的 `effective_config.json`、`model_config` 和 `model_semantics_version` 为准，不能用当前 YAML 反推历史模型。当前代码为 v40；`results/main` 中的旧版本资产不得直接组成最终对照表。

## 2. 训练命令

从零训练：

```powershell
cd E:\paper1\deeplearn\train
python mban.py --config config.yaml --set resume=none --set initial_checkpoint=null
```

继续断点训练：

```powershell
python mban.py --config config.yaml
```

只加载已有权重并重新开始优化：

```powershell
python mban.py --config config.yaml `
  --set resume=weights `
  --set initial_checkpoint=path\best_val.pth
```

每个实验使用独立 `output_directory`。论文主 checkpoint 为 `best_val.pth`；`latest.pth` 只用于续训。

## 3. 模型语义

训练输入是 H5 中已经完成 TOF/相位对齐和 RMS 归一化边界处理的复数 IQ，不是训练时直接读取 RF ADC。

```text
2 × network_channels IQ features
→ FC1: H
→ FC2: H
→ FC3: K controls
→ linear K→M interpolation
→ active-aperture weights
```

现有主线 `network_channels=160`，FC1 输入为 320 维；mapping ledger 的 `physical_channels=192`，代表性像素 active aperture 为 80。H32×OC8 主线矩阵为 `32×320`、`32×64`、`8×64`；Direct-Full 仅把输出层改为物理通道维度。正文须区分 network channels、physical channels 和 active aperture。

E/explicit 会显式生成 M 维物理权重；F/factorized 使用相同插值基直接合成等价 IQ 结果，不物化 M 维中间权重。factorized 不等于减少物理 IQ 通道访问，也不能自动宣称减少一次通道数据遍历。

dual 的正负支路为 `x+=ReLU(x)`、`x-=ReLU(-x)`；正负 rail 使用共同分母，`dual+relu` 不重复执行第二个 ReLU。当前默认不启用 hidden L1/L2；只有 YAML 显式配置时才存在。

## 4. QAT 和量化

- PTQ：FP32 checkpoint 经过 train split 校准后直接评估，不更新权重；
- 普通 QAT：只在量化前向中继续训练；
- HWA-QAT：按 `qat_schedule` 在 forward 中加入 YAML 选定的硬件非理想。

当前 QAT schedule 为 100 epochs：前 5 epochs 使用 `ideal`，后 95 epochs 使用 `common`。observer 校准只使用 train calibration subset；test 数据不能参与校准。投稿主线 QAT 结果当前为 30-run common MC、每次 10 个 test frames。

```powershell
python mban.py --config config.yaml `
  --eval-config hardware_eval.yaml `
  --mode qat --profile common `
  --set resume=none `
  --set initial_checkpoint=path\best_val_deploy.pth
```

不要把“4-bit 权重直接塞进 FP32 软件前向”当作部署结果。部署评估必须使用匹配的输入量化、层间路径、control ADC 和非理想配置。

## 5. 实验与回归入口

从项目根目录执行：

```powershell
python train\eval_core\static_k8.py --help
python train\evaluate_mban.py --mode figures --help
python train\evaluate_mban.py --mode trace --help
python -m unittest discover -s train\tests -p "test_*.py"
```

特殊实验协议属于 `train/eval_core/`；图表和硬件 trace 由 `evaluate_mban.py` 统一调度。训练入口仍是 `mban.py`，评估输出必须写入独立结果目录，并保留 manifest。

## 6. 训练后检查

正式实验前确认：

- split、seed、数据版本和输出目录已记录；
- checkpoint 保存 `checkpoint_identity` 以及完整 `model_config/evaluation_config`；身份只区分 FP32/QAT 训练，不包含 PTQ；
- H32/H64、K 候选不混用 checkpoint；
- QAT profile、随机种子和阶段切换可追溯；
- 未把旧 v21/L1/K16 结果当作当前 YAML 结果；
- 最终对照表中的 checkpoint 语义版本、control normalization、loss 和 deploy-folded 来源完全一致。
