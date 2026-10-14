# Python 审查报告：beamforming_demo

日期：2026-10-14
范围：项目根目录 `algorithms/`、`data/`、`evaluation/`、`tools/`、入口和测试中的全部 20 个 `.py` 文件。审查依据包括 `config.yaml`、`README.md`、`environment.yml`、HDF5 内嵌 `config_yaml` 结构，以及运行、数据校验和评估路径。

## Git 与范围

- 审查开始时 beamforming_demo 工作区干净，位于 `main`；本报告和改动均属于本次审查。
- deeplearn 现有 `train/docs/train_review_20261014.md` 用户改动未触碰、未暂存。
- 两个项目 16 个同路径共享 Python 文件已逐个 SHA256 比对，内容完全一致：7 个算法实现、`run_one.py`、`run_all.py`、评估器与绘图器及其测试、算法审计测试、向导、消融工具。
- `algorithms/template_algorithm.py` 保留有意差异：deeplearn 中确实有 `data/scripts/pack_EPFL.py` 的 teacher 接口；beamforming_demo 中没有该文件，因此只从 beamforming_demo 删除了指向不存在接口的过时说明。
- 两项目的数据打包/校验契约和 HDF5 数据并不相同。beamforming_demo 的 `all_available` 表示完整角度集合；deeplearn 的打包器允许明确选取角度子集，所以没有把 beamforming_demo 的数据校验规则机械复制到 deeplearn。

## Python 文件登记

| 文件 | 检查职责与结果 |
|---|---|
| `algorithms/common.py` | 公共参数、输入采样、孔径/窗函数、协方差、输出和缓存路径；抽出三种 MV 系算法共用的空间/时间协方差与 FBSS。 |
| `algorithms/das.py` | DAS 离散/几何孔径、固定/动态孔径、插值和 TGC 路径；合成分支与真实数据冒烟通过。 |
| `algorithms/mv.py` | MV 空间平滑、FBSS、对角加载和权重求解；共用协方差路径已验证。 |
| `algorithms/esbmv.py` | ESB-MV 特征值数量/阈值分支、加载与权重；共用协方差路径已验证。 |
| `algorithms/gcfmv.py` | GCF-MV 协方差、频率 bin 与 GCF power 分支；共用协方差路径已验证。 |
| `algorithms/cmsaw.py` | CMSAW 子阵列平滑、裁剪和深度平滑；合成分支检查通过。 |
| `algorithms/fdmas.py` | F-DMAS 配对求和与分块执行；合成分支检查通过。 |
| `algorithms/template_algorithm.py` | 尚未实现的脚手架，`beamform()` 按设计抛出 `NotImplementedError`；删除本项目中不存在的 pack-time teacher 说明。 |
| `data/check_data.py` | HDF5/YAML 模式、角度选择、样本元数据和 PICMUS 路径校验；修复两项漏检。 |
| `data/pack_data.py` | 输入样本、角度、IQ/GT 与元数据写入链路；结合数据契约和临时数据打包测试检查，无确认缺陷。 |
| `data/tests/test_data_audit.py` | 加入全角度选择和 PICMUS 路径越界回归用例。 |
| `evaluation/evaluate.py` | 方法/标签解析、GT 元数据、指标、协议报告；修复错标与截短 GT 的静默降级。 |
| `evaluation/plot_metrics.py` | 指标图、重复参考模式推断；`auto` 不再把缺失/损坏元数据猜成模型模式。 |
| `evaluation/tests/test_evaluation_audit.py` | 覆盖方法标签、GT 样本数、模型元数据和自动参考模式；包含 contrast/resolution 合成评估绘图冒烟。 |
| `run_one.py` | 单场景参数转发、算法运行、GT 检查、结果汇总；GT 数据集样本数现在必须与 IQ 一致。 |
| `run_all.py` | 场景循环和单场景入口转发；配置转发与 `run_one` 一致，无确认缺陷。 |
| `tests/test_algorithm_audit.py` | 参数、缓存、算法、向导及运行入口回归测试；新增向导拒绝、GT 截短、消融路径和协方差公式用例。 |
| `tools/run_wizard_cn.py` | HDF5 样本发现、配置选择和算法参数输入；非法默认枚举现在明确报错，样本名不再伪造。 |
| `tools/run_ablation_cn.py` | 消融运行、结果扁平化与评估；删除不可达目录兼容分支和宽泛导入回退。 |
| `tools/ablate_sound_speed.py` | 声速搜索和网格扫描；参数校验、搜索、结果保存路径无确认缺陷。 |

## 配置到行为的追踪

`run_one.load_config()` 读取根 `config.yaml`；`build_algorithm_cmd()` 将通用字段和所选算法字段转成模块 CLI 参数；算法入口合并并校验参数，再传入对应 beamformer。`run_params.json` 记录实际算法、标签、参数和场景，评估器读取该记录确定面板/参考类型。这个项目是成像运行与评估代码，没有训练 optimizer、反向传播或 checkpoint 保存/恢复阶段。

| YAML 字段 | 调用链和实际行为 |
|---|---|
| `algorithms` | `run_one`/`run_all` 选择算法模块；每项最终进入 `algorithms.<name>`。 |
| `algorithm_labels` | 仅用于比较图、运行元数据和评估显示名，不改算法计算。 |
| `params.select_angles` | 传入通用角度选择器，决定参加相干合成的发射角。 |
| `params.f_number`、`dynamic_aperture` | 进入 DAS/MV 系接收孔径计算；动态与固定孔径为不同执行路径。 |
| `params.window` | 进入活动接收孔径加窗；向导默认值必须属于支持的窗口集合。 |
| `params.interp` | 进入 IQ 延迟采样插值；向导默认值必须属于支持的插值集合。 |
| `params.tgc`、`tgc_alpha` | 控制 TGC 开关及增益系数，值由命令行传入算法。 |
| `params.dr` | 控制图像显示/比较的 dB 动态范围，并随评估配置传递。 |
| `algorithm_params.das.aperture_mode` | `discrete` 与 `geometry` 决定 DAS 孔径索引分支；不支持的值在入口失败。 |
| `mv.mv_dl`、`fbss`、`subarray_ratio`、`temporal_win` | 依次影响对角加载、前向/前后向协方差、子阵列尺寸、时间快照数。 |
| `esbmv.mv_dl`、`fbss`、`subarray_ratio`、`temporal_win`、`num_eig`、`eig_threshold` | 公共 MV 协方差配置后，`num_eig > 0` 走指定特征数；`num_eig = 0` 走相对阈值选择。 |
| `gcfmv.mv_dl`、`fbss`、`subarray_ratio`、`temporal_win`、`gcf_low_bins`、`gcf_power` | 公共协方差配置后，GCF bin 截取和 power 控制频域加权。 |
| `cmsaw.mv_dl`、`fbss`、`subarray_ratio`、`temporal_win`、`lmax_ratio`、`min_subarray_len`、`delta_max`、`gamma`、`clip_percentile`、`depth_smooth_rows` | 控制协方差/子阵列、平滑长度、抑制与裁剪、深度平滑。 |
| `fdmas: {}` | 当前配置无额外算法参数，走其默认配对积和分块实现。 |
| `scenes[].id`、`h5_path`、`sample_idx` | 决定输入 HDF5 与样本；IQ 和 GT 的样本数在运行/评估路径检查。 |
| HDF5 内嵌 `config_yaml.input_angle_selection`、`selected_angle_indices` | 声明数据打包选角；本项目声明 `all_available` 时必须选择全部角度且保持顺序。 |
| HDF5 内嵌 `config_yaml.source_samples[].phantom_path` | 作为相对 PICMUS 路径解析，必须仍位于 PICMUS 目录内且文件存在。 |

## 缺陷与修复

### P1：评估标签数量不匹配会静默错标

- 触发：`--methods das,mv --method_labels DAS`，且比较图面板数恰好能通过原有最终数量检查。
- 原顺序：解析方法与标签 → 标签数量没有和方法数量比较 → 后续名称/面板数量检查可能放行 → 图像面板被错误标成 `DAS`，另一个方法名被丢弃。
- 预期：每个方法恰好有一个显示名；独立传标签也应失败。
- 修复：在面板加载前要求 `--method_labels` 必须与 `--methods` 同用且数量一致。

### P1：截短 GT 会被当成无 GT 样本继续评估

- 触发：HDF5 含 `all_envdb_norm`，但 GT 样本数小于 `all_multi_I`，例如 2 个 IQ、1 个 GT，并请求样本 1。
- 原顺序：根据 GT 是否有对应索引设置 `has_gt=False` → 省略 GT 面板/相对指标 → 数据错误变成一条看似合法的无 GT 结果。
- 预期：只要 HDF5 声明 GT，GT 样本数必须与 IQ 一致；否则输入无效。
- 修复：`run_one.scene_has_gt()` 与 `evaluation.read_sample_meta()` 在选择评估模式前验证样本数并报错。

### P1：`all_available` 可与角度子集元数据矛盾

- 触发：`input_angle_selection: all_available`，但 `selected_angle_indices` 例如为 `[0, 2]`，输入有 3 个角度。
- 原顺序：校验角度/索引类型与重复项 → 只验证声明值为 `all_available` → 子集配置仍能通过。
- 结果：数据元数据声称全角度而实际只选子集，造成训练/算法输入不可比，且检查器未拦截。
- 修复：当声明 `all_available` 时要求索引等于完整、原顺序的 `np.arange(angle_count)`。

### P1：PICMUS 模体相对路径可越出允许目录

- 触发：`phantom_path` 使用 `../outside.h5` 且目标文件存在。
- 原顺序：直接拼接 `data/PICMUS / phantom_path` → 只检查 `is_file()` → PICMUS 外的文件可通过。
- 预期/结果：模体引用必须属于 PICMUS 根；否则数据来源定义失效。
- 修复：解析规范路径，先验证其位于 PICMUS 根，再验证文件存在。

### P1：模型运行元数据不完整时会猜测评估基线

- 触发：当前格式 `run_params.json` 缺少 `evaluation` 或 `evaluation.skip_baselines`，或 `auto` 模式下缺少/同时包含 `models` 与 `algorithms`。
- 原顺序：评估器把缺失值替换成空对象/默认布尔值，绘图器对文件缺失、损坏或格式不识别回退到模型模式 → 基线或重复参考模式可能与实际运行不符。
- 修复：当前格式要求 evaluation 对象及布尔 `skip_baselines`；`auto` 必须从有效元数据唯一识别 models 或 algorithms，否则入口报错。

### P2：向导会把不支持的默认值换成第一个选项

- 触发：YAML 的 `window`、`interp` 或 DAS `aperture_mode` 不是支持值，用户直接回车。
- 原顺序：搜索默认索引失败 → 默认取 0 → 向导悄悄改成另一个算法配置。
- 修复：读取默认配置时要求精确匹配，不支持的配置立即 `ValueError`。

### P2：HDF5 样本缺标识时生成虚构名称

- 触发：`config_yaml.source_samples[i]` 没有非空 `id` 和 `acquisition_id`。
- 原顺序：向导用 `sample_i` 替代 → 结果列表把合成序号当来源身份展示。
- 修复：要求真实来源标识并入口失败；不创建兼容名称。

### P2：消融结果查找保留不可达的旧嵌套路径

- 触发：后续汇总从已经成功并扁平化的结果行查找方法文件。
- 原顺序：先尝试 `<scene>/<method>.<suffix>`，不存在时再找 `<scene>/<method>/<method>.<suffix>`；成功结果只保留第一种布局。
- 修复：只保留当前扁平路径。

### P2：宽泛 `ImportError` 会掩盖真实内部导入错误

- 触发：向导模块内部依赖导入失败。
- 原顺序：捕获所有 `ImportError` → 尝试另一个同名模块搜索路径 → 首个错误被误当成启动方式差异。
- 修复：使用项目包导入 `from tools import run_wizard_cn as wizard`，真实缺依赖错误直接显现。

### P2：三个 MV 系实现重复协方差/FBSS 计算

- 触发：维护 MV、ESBMV、GCF-MV 的空间/时间快照或 FBSS 公式。
- 原情况：三个类复制同一 reshape、快照数、协方差与前后向平均代码，后续改动可能产生算法间漂移。
- 修复：抽到 `algorithms.common.spatial_temporal_covariance()`；差异算法保留各自加载、特征投影和 GCF 后处理。

### P2：算法模板文档指向本项目不存在的旧 teacher 接口

- 触发：使用 beamforming_demo 的模板开发新算法。
- 原情况：文档声称 H5 packer 会查找 `BeamformerIQ`，但仓库没有所指的 `data/scripts/pack_EPFL.py`。
- 修复：只从 beamforming_demo 删除失效说明；deeplearn 因该 packer 实际存在，保留其项目专属接口说明。

## 分支矩阵与验证

| 分支/交互 | 核对的预期 | 验证 |
|---|---|---|
| DAS 动态/固定孔径 × discrete/geometry | 支持路径各自使用正确孔径；geometry 不被离散默认覆盖。 | 算法合成分支冒烟；DAS 真实数据端到端冒烟。 |
| DAS 插值、TGC、窗函数开关 | 配置字段传到算法并影响对应处理；非法向导默认不改值。 | 算法分支冒烟与向导拒绝用例。 |
| MV/ESBMV/GCF-MV FBSS 开/关 × temporal window | 协方差快照数、矩阵尺寸和前向/FBSS 公式符合定义。 | 新公式测试对比显式矩阵运算，另有 CPU 算法冒烟。 |
| MV 系对角加载与子阵列尺寸 | 加载只进入协方差求解路径，子阵列长度由配置比例/上下界决定。 | 权重公式、奇异协方差和合成算法冒烟。 |
| ESBMV 指定 `num_eig` / 自动阈值 | `num_eig > 0` 使用指定数量，零值使用 `eig_threshold`。 | 两分支合成输入检查形状和有限值。 |
| GCF `gcf_low_bins` / `gcf_power` | bin 截取与权重幂次分别生效。 | 零/非零 power 和 bin 分支合成冒烟。 |
| CMSAW 子阵列、裁剪与深度平滑参数 | 相关配置进入 CMSAW 路径，不影响其他算法。 | 深度平滑开关和参数分支合成冒烟。 |
| F-DMAS 配对/分块 | 分块不改变完整输出形状，所有结果有限。 | 配对与块分支合成冒烟。 |
| 评估 GT/无 GT、方法标签、models/algorithms auto | 有 GT 时样本数和标签准确；元数据不完整时拒绝猜测。 | 25 项评估测试，包括 contrast/resolution 评估与绘图。 |

未对没有行为差异的字段做笛卡尔积组合。未发现 P0。所有已复现 P1 已修复；审计未留下待修复 P1。

## 测试与冒烟结果

在 `paper1` Conda 环境执行：

- beamforming_demo：`conda run -n paper1 python tests/test_algorithm_audit.py` — **24/24 通过**；包含 DAS 真实入口子进程和协方差/FBSS 公式回归。
- deeplearn：同命令 — **24/24 通过**。
- beamforming_demo：`conda run -n paper1 python data/tests/test_data_audit.py` — **9/9 通过**，包含临时 HDF5 打包/校验。
- deeplearn：`conda run -n paper1 python data/scripts/tests/test_data_audit.py` — **8/8 通过**。
- 两项目：`conda run -n paper1 python evaluation/tests/test_evaluation_audit.py` — 各 **25/25 通过**，含 contrast 与 resolution 合成评估、指标表和绘图。
- 两项目：`conda run -n paper1 python -m compileall -q algorithms data evaluation tools tests run_one.py run_all.py` — **通过**。
- 向导与消融工具的脚本/模块 `--help` 冒烟在两项目均返回 0。
- 真实输入冒烟：两项目各运行 `run_one.py --scene simulation_contrast_speckle --algorithms das --no_evaluate`，输出到唯一临时目录；GPU DAS 均成功，数组为 609×387、全有限、峰值 0 dB。beamforming_demo 范围为 −78.668…0 dB；deeplearn 为 −89.820…0 dB。两份 HDF5 分别是 128 通道/75 角度与 192 通道/1 角度，因此输出不应要求数值相同。临时输出已删除。
- 代理对六种算法（DAS、MV、ESBMV、GCF-MV、CMSAW、F-DMAS）和上述差异分支做 CPU 合成冒烟；输出形状正确、有限且非零。
- `pytest` 在环境中不可用；审计使用仓库现有 `unittest` 文件直接执行。

## 未确认风险

- 一次随机合成输入中，ESBMV `num_eig=1` 峰值约 92.5，而自动阈值分支约 0.92。现有单样本证据不足以证明违反定义，已保留为数值行为风险，没有据此改算法或叠加保护。需要基于固定定义输入和期望幅度进一步评估。

## 结束状态

20 个 Python 文件均已登记；P0/P1 已处理；相关测试、合成分支冒烟、真实 DAS 冒烟和编译通过；本次生成的真实数据冒烟临时目录已删除。beamforming_demo 与 deeplearn 的修改分别提交在 `codex/beamforming-python-audit-20261014` 和 `codex/train-python-review-20261014`。
