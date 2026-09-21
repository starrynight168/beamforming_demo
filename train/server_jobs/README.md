# Server jobs

这里只保留可复现的主入口。所有脚本都支持通过环境变量覆盖项目根目录和 Python：

```bash
PROJECT_ROOT=/path/to/deeplearn PYTHON=/path/to/python \
  bash train/server_jobs/ablations/bias_ablation.sh all
```

入口约定：

| 文件 | 项目 | 参数 |
|---|---|---|
| `ablations/bias_ablation.sh` | bias | `train`、`eval`、`all` |
| `ablations/normalization_orchestrator.py` | hidden normalization orchestrator | `--stage {fp32,qat,qat_linf,train,eval,all}` |
| `ablations/loss_ablation.sh` | loss regularizers (d1, d2, lrange) | `fp32`、`qat4`、`qat4_linf`、`train`、`eval`、`all` |
| `ablations/outlimit_ablation.sh` | output constraint | `train`、`eval`、`all` |
| `ablations/windows_ablation.sh` | K8 / direct-full windows | `train`、`eval`、`all` |
| `ablations/model_ablation.sh` | hidden-width × output-control matrix | `fp32`、`qat4`、`all` |
| `ablations/control_adc_ablation.sh` | control ADC range/normalization matrix | `fp32`、`qat4`、`eval`、`all` |
| `ablations/branch_activation_orchestrator.py` | activation × branch matrix | 直接执行；`--eval-gpu` 可覆盖评估卡 |
| `ablations/envelope_orchestrator.py` | envelope loss weight sweep | 直接执行；`--eval-gpu` 可覆盖评估卡 |
| `ablations/envelope_sweep.sh` | envelope sweep wrapper | 直接执行 |
| `diagnostics/hardware_trace.sh` | hardware trace | `bias`、`windows` |
| `run_bias_interlayer_pipeline.sh` | ordinary/array × inter-layer 全流程 | 直接执行 |
| `run_lrange_pipeline.sh` | Lrange FP32/QAT 消融全流程 | 直接执行 |

> **控制归一化规范**：统一全流程基准采用 `control_normalization=none`（与模拟/硬件实际相符，杜绝边缘靶点振荡），仅在显式对比消融阶段（如 `QATlinf`、`QAT4linf`）启用 `linf`。

默认读取 `train/results/SI/shared_split_seed42_0p7-0p2-0p1.csv`，不存在时回退到
`train/results/model_ablation/shared_split_seed42_0p7-0p2-0p1.csv`。GPU 可用脚本专属环境变量覆盖，见各脚本开头。

两个主流程入口也统一放在本目录；它们复用 `lib/common.sh`，支持 `PROJECT_ROOT`、`PYTHON`、`SPLIT_FILE`、`DATA_H5` 等环境变量覆盖。

旧脚本是历史运行副本，已移除；结果目录不在本次整理范围内。
