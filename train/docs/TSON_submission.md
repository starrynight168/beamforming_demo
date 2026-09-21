# TSON 投稿结果登记

本文件是基于 `train/results/` 的投稿口径登记，不替代 `config.yaml`、`hardware_eval.yaml` 或 checkpoint 元数据。最终正文数值只从同一语义版本、同一配置和同一评估协议的 CSV/JSON 读取。

## 当前状态

结果资产已生成，但当前 `results/main` 还不能直接作为最终稿定稿：主线 checkpoint 存在语义版本和配置混用，需统一后复跑主线评估并重生成 Fig. 3–10。

### 已确认的主线资产

- 固定 split：`shared_split_seed42_0p7-0p2-0p1.csv`，344 train / 98 validation / 50 test。
- 当前 `main` 活体报告使用固定 test split 中 10 帧；不能把它写成 50 帧活体结果。
- 四个受控场景：simulation/experiment × contrast/speckle、resolution/distorsion。
- PTQ ladder：3/4/6/8 bit，`none` 与 `linf`；PTQ4 `linf` 是 Fig. 8/9 的下游入口。
- common 压力与 HWA-QAT 评估均为 MC30；QAT schedule 为 5 epoch `ideal` + 95 epoch `common`。
- Fig. 1–10 均有 `figure_manifest.json`；Fig. 10 是 64×64 mapping/resource ledger，不是完整芯片 PPA。

### 主线一致性审计

| 资产 | 语义版本 | 关键配置 | 投稿用途 |
|---|---:|---|---|
| `main/model/FP32/MBAN/best_val.pth` | 34 | H32×OC8，`control_normalization=linf`，`lrange=0.01` | 当前 MBAN FP32 结果，但与其他主线不一致 |
| `main/model/FP32/Direct160/best_val.pth` | 31 | Direct160，`control_normalization=none`，`lrange=0.001` | 当前 Direct 基线 |
| `main/model/FP32/Project160/best_val.pth` | 31 | projection K=8，`control_normalization=none`，`lrange=0.001` | 当前 projection 对照 |
| `main/model/QAT4/MBAN/best_val.pth` | 31 | H32×OC8，`control_normalization=none` | 当前 QAT 结果，但未与 v34 FP32 对齐 |
| `main/model/deploy_folded/MBAN_deploy.pth` | 字段为 31；hash 对应 v34 MBAN | 由 v34 MBAN 权重折叠得到，`derived_from` 在当前工作区不可解析 | PTQ/mapping 入口，需重建元数据 |

因此，当前图包中的 FP32/PTQ4/HWA-QAT 数值只能作为结果诊断表，不能直接表述为严格同配置的 recovery 实验。

## 当前资产诊断数值

以下为现有 CSV 中的活体 `SSIM_dB` 均值，仅用于定位结果和重跑前记录：FP32 为 10 帧；PTQ4 `linf` 与 QAT4 `common` 为 30 次 realization × 10 帧。

| 候选 | FP32 | PTQ4 `linf` | HWA-QAT4 `common` |
|---|---:|---:|---:|
| MBAN | 0.9735 | 0.6056 | 0.9447 |
| Project160 | 0.9472 | 0.7883 | 0.8743 |
| Direct160 | 0.9264 | 0.6754 | 0.8211 |

来源分别为 `main/evaluation/FP32/invivo/test_ssim_summary.csv`、`main/evaluation/PTQ_ladder/PTQ4_linf/invivo/test_mc_summary.csv` 和 `main/evaluation/QAT4/invivo/test_mc_summary.csv`。当前数值支持的保守观察是：MBAN 在 FP32 与 HWA-QAT4 common 中优于两个学习对照，HWA-QAT 相比 PTQ4 `linf` 有明显恢复；PTQ4 `linf` 的 MBAN 结果仍明显退化，不能写成“完全恢复”或“无损 4-bit 部署”。

## 统一后应冻结的 TSON 主线

- 主模型：MBAN，H32×OC8，2 hidden layers，dual，ReLU，RMS input，hidden normalization none，nonnegative hard unity，linear K→M，dynamic aperture，`f_number=1.5`。
- 对照：Direct160 与 Project160；三者必须使用同一语义版本、同一 split、同一训练/评估配置和同一统计协议。
- 维度口径：网络输入为 160 个 IQ 通道，即 FC1 输入 320 维；mapping ledger 的 physical channel dimension 为 192；代表性像素记录的 active aperture 为 80。正文必须区分 `network_channels`、`physical_channels` 和 active aperture。
- 硬件压力：`common` 是文献标定的行为级组合压力，含 64×64 IR-drop、4-bit conversion/noise、write/read variation、drift snapshot、stuck-at 和 TIA/reference 项；不代表真实器件 worst-case。
- mapping：64×64、parallel conversion、post-TIA differential subtraction，分别记录 analog/digital inter-layer；只报告 logical coefficients、differential devices、tiles、converter counts 和 ledger。

## TSON 正文可声明与不可声明

可声明：

- MBAN 将自适应 apodization 从全物理通道输出重参数化为低维 K-control；
- native K-control 与 Direct/projection 的质量—资源差异；
- 在明确的 behavioral common profile 下，4-bit QAT 对 PTQ 退化的恢复趋势；
- 64×64 阵列映射和分项资源账本。

不可声明：

- 真实 RRAM 芯片实测、retention law 或 universally worst-case；
- probe-to-B-mode 全模拟实现；
- 动态逐像素电导写入已经完成硬件验证；
- `inference_seconds` 等同芯片 latency/energy；
- mapping ledger 等同完整芯片面积、功耗或端到端 PPA。

## 投稿前唯一阻塞项

- [ ] 选择并统一语义版本（建议按当前代码 v40 重训/重评全部主线候选，或完整冻结 v31 运行环境）。
- [ ] 用同一来源 checkpoint 重建 `deploy_folded`，修复 MBAN 的版本字段与失效 `derived_from` 路径。
- [ ] 在统一配置下重跑 FP32、PTQ ladder、single-factor、QAT4 和 mapping 相关主线输出。
- [ ] 重生成 Fig. 3–10 及 manifest，并只把统一后 CSV/JSON 的数值写入正文。
