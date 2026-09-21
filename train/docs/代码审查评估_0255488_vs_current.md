# `0255488` 基线与当前版本代码审查评估

> 历史审查快照：本文记录的是早期语义版本 26 的代码审查，不是当前投稿结果依据。该审查时的代码为 v26；当前代码为 v40。主线结果的版本/配置审计与 TSON 投稿口径见 [`TSON_submission.md`](TSON_submission.md)。

审查日期：2026-09-02
基线：`0255488`
当前版本：本分支 HEAD（`codex/pre-rollback-3d10fe1`）

## 1. 审查范围与结论

本报告只比较基线到当前版本实际提交的代码，不把此前的口头评估直接当成结论。重点检查：

- 输出约束是否已经正确解耦；
- `factorized` 是否真正省略了 `M` 维权重物化；
- `projection_controls` 的方向、顺序和一致性；
- 组合式 loss 是否正确计权；
- E/F 对训练、成像和权重诊断的职责边界；
- 配置、checkpoint、Wizard 和评估协议是否被无意改变。

当前本机验证结果：

- `python -m unittest discover -s train -p 'test*.py'`：41 项通过；
- 本机 CUDA 可用，包含 batch>1 的 factorized CUDA 回归测试；
- `python -m compileall -q train`：通过；
- `ruff check train`：通过；
- CPU/CUDA factorized benchmark 和 E/explicit 训练 forward/backward 冒烟：通过；
- `algorithms/mv.py` 与 `0255488`：字节级无差异；
- 当前工作区干净。

总体结论：当前版本保留组合式配置、loss 分解、E/F 控制域等价路径和工程校验。E/explicit 是训练与权重诊断路径；F/factorized 只用于直接 output controls 的部署成像，不生成 M 维权重；projection 保持 E/explicit。评估入口按当前约定执行。

## 2. `0255488` 到当前版本的实际变化

| 模块 | 基线 | 当前版本 | 判断 |
|---|---|---|---|
| 输出约束 | `output_constraint` 复合枚举 | `output_domain + unity_constraint` | 方向正确，删除旧字段后应同步升级语义版本 |
| Loss | `iq_weight`、`envelope_weight`、单一 aperture regularization | `loss.terms.iq/envelope/unity/d1/d2/lrange` | 方向正确，D1/D2 已可独立并行 |
| 插值 | factorized 主要固定为 linear | 支持 linear/nearest/cubic；固定路径使用二维基矩阵，动态路径使用索引和 `scatter_add_` | 已增加基矩阵与 resize 一致性测试 |
| factorized | basis 后使用高维 `einsum` | 固定孔径使用二维 `matmul`，动态孔径使用控制域累加 | 避开高维 batched-CUBLAS；速度和显存以实测为准 |
| E/F 权重职责 | 大多数路径直接生成 M 维权重 | E 始终生成，F 直接路径始终不生成 | 不再由调用方传权重生成开关 |
| ADC | 旧有 observer/signed 路径 | `observer` 或 `full_scale`；固定量程的符号范围由 output_domain 决定 | 量程策略与输出值域解耦 |
| 配置列表 | 基本检查 | 非空、类型、重复、未知层、数值排序 | 正确且低风险 |
| projection | 原有显式 M→Kp→M | 继续由 E/explicit 执行；F/factorized 直接路径拒绝 projection | 避免把 projection 与 F 混成另一套未验证语义 |
| 评估 | 包含 scenes、CIRS、all、test 等路径 | 删除直接 CIRS/all/simulation 模式，统一固定帧 test | 改变评估协议，不是单纯代码优化 |
| 计时 | CUDA Event 使用默认流 | 绑定目标设备当前流 | 正确的独立工程修复 |
| checkpoint | 版本值为 25 | 新字段使用语义版本 26，旧版本明确拒绝 | 保护不同语义模型不混用 |

## 3. F 流程的真实数据流

设：

- `M=160`：物理通道数；
- `K`：网络原生输出控制点数，例如 8；
- `Kp`：后置 projection 控制点数。

### 3.1 当前 factorized、无 projection

当前路径已经接近目标：

```text
raw controls[K]
→ 输出域/ADC
→ 插值索引
→ 将 IQ 按索引累加到 K 个 basis
→ controls[K] × basis[K]
→ 波束输出
```

它不需要先生成完整的 `weights[...,160]`。当前实现使用 `scatter_add_`，没有再使用此前容易触发服务器 batched-CUBLAS 的高维 `einsum`。

### 3.2 explicit 路径

```text
raw controls[K]
→ 插值到 M=160
→ 与 IQ 的 160 个通道逐元素相乘
→ 通道求和
```

这是参考实现，便于检查数值和梯度。

### 3.3 当前 projection 路径

```text
raw controls[K]
→ 插值到 M
→ M→Kp
→ Kp→M
→ 与 IQ 相乘
```

因此 `projection_controls` 的最终输出仍然是 M 个物理通道，并没有把 IQ 的物理通道数从 160 变成 Kp。Kp 的作用是压缩和约束权重曲线的自由度。

当前代码按实现方式固定权重职责：

```text
E/explicit：K controls → M 维权重 → IQ[M] 合成
F/factorized：IQ[M] → 插值基的转置作用 → K 维等效 IQ → controls 合成
```

F 直接路径不生成 M 维权重；动态孔径和 linear/nearest/cubic 插值仍在 F 中执行。F 不支持 projection，projection 使用 E/explicit。

## 4. 已确认可以保留的改动

### 4.1 输出约束正交解耦

当前把原来的：

```text
relu_hard / relu_free / signed_hard / signed_free
```

拆成：

```text
output_domain: nonnegative / signed
unity_constraint: hard / free
```

这个拆分是合理的，因为输出值域和 unity 约束确实是两个不同概念。配置、模型初始化、训练日志、checkpoint metadata 已基本同步。

旧 checkpoint 现在会因 `model_semantics_version` 不匹配而明确拒绝；旧字段也不再由配置加载器兼容。

### 4.2 组合式 loss

当前代码已经支持：

```text
IQ + envelope + unity + D1 + D2 + Lrange
```

其中 D1 和 D2 可以同时启用，符合组合式设计。`loss.terms` 的权重也已经在配置校验和训练损失中分别读取。

保留该设计；当前训练路径已在 D1/D2 启用时自动请求权重物化。

- 日志显示的 loss 是否始终是“原始项”和“加权项”的同一口径；
- `absolute` 域与 hard unity 的限制是否过于严格。

### 4.3 layer list 校验

当前增加了：

- 列表类型检查；
- 非空字符串检查；
- 重复层名检查；
- hidden layer 范围检查；
- 按 `fc1、fc2、...` 数字顺序规范化。

这是无损规范化，可以保留。`weight_transform=ws` 时空列表自动扩展为所有有效层属于显式默认行为，但 Wizard 中其他自动删改仍需处理。

### 4.4 control-space factorized

当前 factorized 的主要改动不再依赖高维 `einsum`：固定孔径用二维矩阵乘，动态孔径用控制点维度累加 IQ。这个方向适合 F 的目标：减少 M 维权重中间量，而不是声称减少物理器件数或必然减少 MAC。

准确表述应是：

- `factorized` 减少软件侧 M 维权重物化、缓存和搬运；
- `output_controls=K` 才决定原生输出层的控制点/阵列规模；
- 同一 H、同一 K 下，factorized 本身不等于减少 CIM 器件；
- 速度和 MAC 必须由实际 benchmark 决定。

本机 CUDA 轻量基准（M=160、K=8、FP32）表明，速度和峰值显存会随孔径类型、插值方式和实现路径变化，不能用一个固定倍率概括 factorized 的收益；正式结论应使用相同参数分别测量。服务器结果仍需单独测量。

### 4.5 CUDA Event 修复

当前评估计时把 CUDA Event 记录到目标设备的当前流，解决了 `cuda:7` 进程使用默认设备流造成的“Event 尚未完成”假错误。这是独立、低风险、应保留的工程修复。

## 5. 当前仍存在的逻辑风险

### 5.1 checkpoint 语义版本已升级

当时审查的源码是：

```text
MODEL_SEMANTICS_VERSION = 26
```

checkpoint 的字段已经从：

```text
output_constraint
iq_weight
envelope_weight
regularization
```

变成：

```text
output_domain
unity_constraint
loss
```

这不是同一语义版本。当时的 `MODEL_SEMANTICS_VERSION=26`，旧 v25 checkpoint 会在训练/评估加载阶段明确拒绝；配置加载器也不再接受旧字段。当前版本请以运行时代码和 checkpoint 元数据为准。

### 5.2 fixed projection 与 signed-hard 的顺序已修复

固定孔径物化路径当前为：

```text
candidate weights → project_output_weights → mask_weights_to_aperture
```

固定 signed-hard 的 unity 已覆盖支持的插值路径；projection 仍使用显式路径。

### 5.3 E/F 权重职责边界

训练入口固定使用 E/explicit，因此 D1/D2 可以作用于插值后的 M 维物理权重。普通 F 部署成像只返回 controls 并直接合成，不生成 M 维权重。权重曲线、权重统计和权重诊断统一使用 E/explicit；代码不再向调用方暴露权重生成开关。

### 5.4 `control_normalization=none` 的零向量行为已冻结

`mode=none` 对有效的非零控制值不做缩放。遇到全零或无效参考值时，统一退化为当前有效孔径上的均匀控制值，即 DAS 矩形窗，并通过 `hardware.adaptive_valid` 标记本次 fallback。

这是自适应权重失效时的明确工程规则，不是把正常的 `none` 重新解释成归一化；已有全零控制和 `adaptive_valid` 测试覆盖。

### 5.5 `absolute` D1/D2 的 validator 可能过于严格

当前对以下组合直接报错：

```text
nonnegative + hard + D1/D2 absolute
```

这是当前 loss 语义的明确约束：nonnegative+hard 下硬 unity 已固定整体尺度，absolute D1/D2 不作为当前有效实验组合；不属于运行时 bug，本次不修改 loss 规则。

### 5.6 当前测试覆盖范围与剩余验证

已有 41 项测试覆盖了控制归一化、输出组合、factorized fixed/dynamic、梯度、显式 projection、signed-hard unity、D1+D2、CUDA batch>1 和批量 cubic projection。E/explicit 训练 forward/backward 冒烟已通过，F 的无权重执行路径也已覆盖。仍建议：

- 真实 H5 的最小 CUDA batch 训练/评估；
- 服务器目标 GPU 上按 benchmark 脚本复测时间和峰值显存。

## 6. 评估器与文档的变化

### 6.1 评估协议被改变，不应和模型代码变化混合

当前 diff 删除了：

- `cirs` 模式；
- `all` 模式；
- 独立 simulation reference-only 输出；
- CIRS 随机帧参数。

同时新增了从固定 split 中确定性选择 `test-frames` 的流程，并统一使用 `all_envdb_norm`。

固定 split 本身是有价值的，但这属于评估协议变化，不是 factorized 修复。它会改变旧结果的：

- reference；
- 帧集合；
- CSV 字段；
- SSIM 统计；
- 模型排名。

因此当前固定 test-split、reference 和帧选择应作为当前评估协议单独记录，不能与基线结果混列；CIRS/all 的删除属于已确定的入口变化，不应在模型算法结论中重复解释。

### 6.2 文档存在明显漂移

当前源码和顶层说明已经是语义版本 26；但以下文档包含历史实验记录，仍出现旧版本、旧字段或旧主线描述：

- `train/docs/MBAN.txt`；
- `train/docs/README_TRAIN.md`；
- `train/docs/README_VALIDATE.md`；
- `train/docs/实验资产清单_主线与消融_v30.md`；
- `train/docs/FIG流程_主线硬件压力_v31.md`。

`MBAN.txt` 已在前言声明旧 v21–v30 段落为历史记录；`实验资产清单_主线与消融_v30.md`、`FIG流程_主线硬件压力_v31.md` 和论文草稿仍需按该声明阅读。尤其是其中的 `joint L1@fc2` 不能覆盖当前 `train/config.yaml` 的 `normalization=none`。这属于文档口径清理，不改变代码和既有实验资产；正式结果应以 `effective_config.json`、checkpoint metadata 和当前配置为准。

## 7. 建议吸收与暂不吸收

### 7.1 可以吸收

1. 输出域和 unity 解耦；
2. `loss.terms` 组合式设计；
3. D1/D2 独立并行；
4. layer list 无损规范化；
5. control-space factorized 的基本数学形式；
6. CUDA Event 目标流修复；
7. checkpoint 的 semantic hash、有效配置和 geometry metadata；
8. fixed/dynamic explicit-factorized 一致性测试框架。

### 7.2 目前不直接吸收

1. 重新换成高维 batched `einsum`；
2. 立即把所有调用方改成任意传播 `weights=None`；
3. 用新 signed/Linf 定义覆盖旧 QAT 语义；
4. 同时重构 ADC、噪声、共模和 MC；
5. 修改 MV/DAS；
6. 中途改变 reference、帧选择和指标公式；
7. 用未经测试的 QAT 兼容矩阵硬编码大量禁止项；
8. 扩大模型候选范围。

## 8. 推荐的后续实施顺序

每一步都执行轻量测试、提交和推送，不能跨步混改。

### 已完成：修 semantic version

- 新字段 schema 使用语义版本 26；
- 明确拒绝旧版本 checkpoint；
- 已加入旧版本拒绝测试。

### 已完成：决定 projection 支持层级

- `factorized + projection` 直接报错；
- projection 统一由 E/explicit 执行；
- E/F 直接路径和 signed-hard 已有数值测试。

### 已完成：决定 E/F 权重职责

- E/explicit 生成并返回 M 维权重；
- F/factorized 直接路径返回 `weights=None` 并完成控制域合成；
- D1/D2、权重统计和权重诊断使用 E/explicit；
- 当前没有通用 `apply/materialize` 接口或自动物化机制。

### 已完成：修 Wizard

- 合法配置只做排序和类型规范化；
- 非法配置直接报错；
- 不再自动删层、关闭归一化或把 complex 改成其他输出域。

### 已完成：benchmark 决定 kernel

- CPU/CUDA；
- batch=1 和 batch>1；
- fixed/dynamic aperture；
- linear/nearest/cubic；
- 时间、峰值显存、数值误差和梯度误差已完成本机轻量验证；服务器需按同一脚本复测。

### 第六步：真实数据与目标服务器验证

signed/Linf、ADC 后端、QAT 兼容性和评估协议当前按已有代码口径保持不变，不在本次审查中重新定义。下一步只需用真实 H5 和目标服务器完成最小 CUDA 验证，不把未验证的性能数字写成正式结论。

## 9. 当前可执行结论

当前版本可以作为“组合式重构开发基线”，但不能把 factorized 描述为减少器件数或必然减少 MAC。代码层已完成语义版本保护、E/F 权重职责划分、Wizard 非静默校验、CUDA/梯度回归和批量 cubic projection 修复；正式软件消融前仍应完成真实 H5 的最小 CUDA 训练/评估冒烟，并继续把评估协议与模型算子分开记录。
