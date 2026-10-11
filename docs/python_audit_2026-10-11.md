# Python 审查记录（2026-10-11）

## 范围与行为约定

- `beamforming_demo`：20 个项目自有 Python 文件。
- `deeplearn`：28 个项目自有 Python 文件；排除 `train/`、`external/` 和第三方 PICMUS 源码。
- 17 个同路径文件核对并同步，共 31 份不同源码。以下清单覆盖函数、类方法、调用方及状态变化；测试文件也审查导入方式、夹具与断言。
- 算法承诺支持当前 H5 的 `[N,A,T,C]` IQ、有效时间长度、角度、物理参数和递增网格；GT 采用 H5 内记录的生成动态范围。脚本按单个进程一次配置运行，不承诺同进程并行修改模块参数。
- PICMUS 定量评估依赖 H5 模体类型、官方定义及完整坐标覆盖；明确的非 PICMUS 数据跳过模体指标。错误声明直接报错，不从 GT 猜测目标。
- 保留 MV 奇异协方差的伪逆计算；没有添加重试、旧格式猜测或自动切换模式。
- MBAN 本轮仅审查根目录适配入口；训练内部、实际 checkpoint 恢复和推理不属于本轮验证范围。

## 逐文件登记

### 两项目共同文件

| 文件 | 函数与调用链检查重点 | 结论 |
| --- | --- | --- |
| `algorithms/common.py` | 公共参数校验、H5 裁剪、角度选择、孔径、窗、插值、MV 求解、输出和缓存 | 修复 GT 解码和 CLI 文件依赖；同步相对 H5 路径修复 |
| `algorithms/das.py` | 离散/几何孔径、插值边界、有效权重、IQ 相位与归一化 | 数值验证通过；同步直接启动修复 |
| `algorithms/mv.py` | 行孔径、子阵列、时间平均、FBSS、加载、协方差和权重 | 数值及求解回归通过；同步直接启动修复 |
| `algorithms/esbmv.py` | MV 协方差、Hermitian 特征分解、自动/固定子空间、CPU/GPU 分工 | 自动及固定子空间冒烟通过；同步直接启动修复 |
| `algorithms/gcfmv.py` | 空间频率选择、功率比和 MV 后加权 | `power=0` 等于 MV，`power=2` 改变输出；同步直接启动修复 |
| `algorithms/cmsaw.py` | MV 依赖、子阵列、相干权重、对角缩放、裁剪、深度平滑 | 支持的默认/边界组合冒烟通过；同步直接启动修复 |
| `algorithms/fdmas.py` | 延时采样、成对乘积、带符号平方根、幅度与 TGC | 默认及 TGC 关闭冒烟通过；同步直接启动修复 |
| `algorithms/template_algorithm.py` | 输入/输出约定和新增算法接入 | 同步直接启动修复；未增加模板算法实现 |
| `run_one.py` | 配置→CLI→子进程→GT→对比→评估→缓存记录 | 修复公共 CLI 参数未贯通和 GT 动态范围错误 |
| `run_all.py` | 场景选择、参数透传、子进程失败传播 | 未确认新的缺陷 |
| `evaluation/evaluate.py` | H5 协议资格、模体/坐标、方法名、显示域与靶标指标、报告 | 修复普通算法元数据无法自动读取的方法名问题 |
| `evaluation/plot_metrics.py` | CSV/JSON、方法顺序、分组、曲线选择、分页及已有输出 | 评估与绘图回归通过；未确认新的缺陷 |
| `tools/run_wizard_cn.py` | 输入发现、H5 资格检查、交互状态、参数生成、运行命令 | 配置生成及资格回归通过；未确认新的缺陷 |
| `tools/run_ablation_cn.py` | 参数取值/重复校验、缓存、单变量配置、结果展开、GT、统一评估 | 修复重新运行后仍沿用旧 GT |
| `tools/ablate_sound_speed.py` | 输入、GT 显示域、MV 重建、评分缓存、区间收缩、密集网格 | 修复两段搜索不收缩和 GT 编码口径 |
| `tests/test_algorithm_audit.py` | 算法边界、缓存、入口、消融和新增问题回归 | 支持所在目录直接运行，新增问题回归通过 |
| `evaluation/tests/test_evaluation_audit.py` | 四类模体、非法协议、数值指标及绘图夹具 | 支持所在目录直接运行，补充算法元数据回归 |

### beamforming_demo 独有文件

| 文件 | 检查重点 | 结论 |
| --- | --- | --- |
| `data/pack_data.py` | 官方 GT/多角度 DAS、物理元数据、RMS、Float16、时间 padding、自描述配置及写入 | 两种 GT 来源打包/检查测试通过；未确认新的缺陷 |
| `data/check_data.py` | 当前 schema、元数据、频率、形状、padding、GT、模体路径及检查报告 | 现有回归通过；未确认新的缺陷 |
| `data/tests/test_data_audit.py` | 打包夹具、插值端点、网格、序列编码、模体类型/路径 | 8 项通过 |

### deeplearn 独有文件

| 文件 | 检查重点 | 结论 |
| --- | --- | --- |
| `algorithms/mban.py` | 公共参数、外部依赖接入、路径、模型类型、PTQ 校准调用、输出 | 修复直接启动导入；内部训练调用未执行 |
| `data/checks/check_data.py` | schema、嵌入配置、块检查、尺度、padding、凭证与报告 | 删除已经被当前 schema 排除的旧元数据重复检查 |
| `data/checks/check_epfl_npz.py` | ZIP 成员、NPY 头、长度、dtype、CRC、修复写入顺序 | 修复小数 shape 被截断且放行 |
| `data/checks/check_single_angle_scale.py` | 中心角、原始 RF 解调、低通、RMS/峰值与 H5 尺度统计 | 与实际打包链路核对；未确认新的缺陷 |
| `data/scripts/common.py` | 复数归一化、压缩配置、紧凑序列、日志和表格 | 现有回归通过；未确认新的缺陷 |
| `data/scripts/download_epfl.py` | 文件选择、范围请求、部分文件、CRC、提交、人工中断及控制台状态 | 6 项中断回归通过；未进行在线下载 |
| `data/scripts/pack_EPFL.py` | YAML 校验、输入角、解调/降采样、teacher 输出、尺度、签名、追加标记和压缩 | 配置/签名/插值回归通过；未重打包完整数据集 |
| `data/scripts/simulate_PICMUS.py` | 固定仿真定义、RF/IQ、重采样、噪声、DAS GT、schema 和写入后校验 | 静态流程核对及入口通过；未重新生成完整仿真 |
| `data/scripts/tests/test_data_audit.py` | 数据块、配置、压缩、NPZ CRC/截断及头格式 | 支持所在目录直接运行，新增 NPY 头回归 |
| `data/scripts/tests/test_download_epfl_interrupt.py` | watchdog、阻塞输出、强制退出、Quick Edit 保存恢复 | 6 项通过 |
| `evaluation/tests/test_runtime_regressions.py` | 缓存依赖、公共配置、评估资格、打包参数/签名 | 支持所在目录直接运行，20 项通过 |

## 配置链路

| 字段 | 解析、执行与结果记录 |
| --- | --- |
| 根配置 `algorithms`、`algorithm_labels` | 入口选择→对应脚本/标签→拼接顺序→`run_params.json`→自动评估方法名 |
| `scenes.id/h5_path/sample_idx` | 场景选择→项目根目录 H5 路径→对应样本/有效长度→GT/模体协议→输出目录 |
| `params.select_angles` | 公共校验→算法角度索引→发射/接收延时与合成；单角选择最接近零角 |
| `f_number/dynamic_aperture/window/interp` | 公共校验→孔径与窗权重→插值采样/边界→算法 IQ 输出 |
| `tgc/tgc_alpha/dr` | 公共校验→TGC 与显示；根入口 CLI 覆盖同步写入实际配置，图像和评估使用同一口径 |
| `algorithm_params` | 算法 CLI→构造参数→协方差/子空间/相干权重；缓存同时记录参数、源码和输入依赖 |
| H5 `generation.ground_truth.dynamic_range_db` | 数据生成→归一化 GT 保存→读取时还原 dB→按本次显示范围裁剪和评估 |
| H5 `source_samples` 中模体字段 | 数据生成→嵌入配置→评估资格/官方定义及坐标检查→对应指标 |
| EPFL `reconstruction/algorithm_params/outputs` | `load_pack_config`→实际 teacher/角度配置→重建→IQ/显示域/幅度/尺度数据集及生成签名 |
| EPFL `pack.input_angles/decimation_factor/norm_mode/max_files/preview_count/preview_root` | 配置校验→角度/时间轴/复数尺度→处理范围/预览→H5 元数据；追加时核对生成签名 |
| EPFL `compression` | 类型与级别校验→H5 创建参数；不默默替换不支持的 codec/level |
| EPFL `datasets` | enabled、输入、输出、样本数量、模体声明→输入收集/处理→样本元数据与数量检查 |

本轮没有验证训练前向、反向、优化器阶段及训练 checkpoint 恢复；这些属于明确排除的 `train/`。MBAN 外部接口未据此宣称内部参数全部生效。

## 已确认问题及最小修改

| 级别 | 触发条件与实际顺序 | 预期、影响及修改 |
| --- | --- | --- |
| P0 | deeplearn 从 `algorithms/` 直接运行脚本：先导入 `algorithms.common`，项目根目录尚未在搜索路径 | 应正常进入入口；原来立即 `ModuleNotFoundError`。同步 demo 已有入口修复，另补 MBAN |
| P1 | 根入口传 `--dr 40 --no_tgc`：原先只追加到算法命令，父进程仍使用 YAML 的 60/true | 重建、图像、评估和缓存应采用实际值；改为先解析公共覆盖，再统一生成命令和记录 |
| P1 | GT 按 60 dB 生成，显示改为 40 dB：原先先把归一化 GT 乘 40 再做对比/指标 | GT 中 0.5 应为 -30 dB，原来错误变为 -20 dB。统一读取 H5 生成范围，恢复物理 dB 后再显示 |
| P1 | `--keep_existing` 配合 CLI checkpoint/config 文件：原先记录参数文本，但未统计 CLI 文件依赖 | 同一路径文件变化应使缓存失效；把 CLI 路径合并到既有依赖签名。相对 H5 缓存路径修复同时同步到 deeplearn |
| P1 | 声速搜索 `sections=2`，最优点位于中心：左右相邻点就是原始两端，下一轮仍使用相同区间 | 搜索区间应收缩；复现八轮不收缩后改为入口明确要求至少 3，不暗改参数 |
| P1 | 已有消融 `ground_truth.npy`，重新运行生成新的 comparison：原先看到旧 GT 就提前返回 | 当前算法结果应配当前 GT；删除该提前返回，刷新 GT |
| P1 | 普通算法 `run_params.json` 后独立使用 `--methods auto`：原先只读取 `models`，直接报缺少字段 | 应支持当前生产者格式；读取 `algorithms/algorithm_labels`，保留既有模型格式 |
| P1 | NPY shape 头含 `(1.5,2,3,4)`：检查器先 `int()`，放行 `(1,2,3,4)`；NumPy 实际加载拒绝 | 检查结论应与加载约定一致；要求非负整数元组和布尔 `fortran_order`，不做隐式转换 |
| P2 | 部分测试从所在目录直接运行，导入项目包时根目录缺失 | 与模块运行保持相同入口；仅补项目根路径 |
| P2 | 当前 schema 已拒绝根级 `gt_fs/decimation_factor`，检查器仍读取它们并重复核对 | 有效配置由嵌入元数据校验；删掉不可用于当前 schema 的旧重复分支 |

## 分支与验证

| 分支 | 预期与实际验证 |
| --- | --- |
| DAS 离散/几何孔径 | 分别核对通道数定义/几何覆盖；两种模式各跑 3 组窗和插值 |
| 窗函数 | rect、tukey、hann、hamming、blackman、kaiser 全部参与数值运行；两通道有效权重另有回归 |
| 插值 | nearest、linear、cubic、quintic、farrow、sinc 全部参与数值运行；端点和越界值另有回归 |
| MV 加载/FBSS/时间平均/孔径 | 默认动态+FBSS+时间窗3，与固定孔径+无FBSS+时间窗1+加载0.01 均运行；奇异与正常求解有公式断言 |
| ESBMV 子空间 | 自动选择和固定一个特征向量均运行；有限输出、形状、峰值归一化通过 |
| GCF | `power=0` 与相同设置 MV 数值一致；`power=2` 输出确实变化 |
| CMSAW | 默认，以及 gamma=1、delta_max=0、平滑3行交互运行；gamma=0 明确拒绝，未放宽定义 |
| F-DMAS/TGC | 默认，与无 TGC+kaiser 窗均运行；有限输出和动态范围通过 |
| PICMUS/非PICMUS | 四协议不依赖样本名；不完整声明/官方定义/坐标覆盖拒绝；非PICMUS跳过；指标 CSV 与绘图冒烟通过 |
| 打包/修复/追加 | demo 两种 GT 来源临时打包；EPFL 当前配置、teacher 参数、生成签名、CRC 和截断回归；完整追加与在线下载未执行 |

16 组 CUDA 数值运行使用临时合成 H5，输出均为 `(8,7)`、有限、最大值 0 dB；最小值介于 -24.53 和 -93.53 dB。小网格配长标题出现一次 tight-layout 警告，不影响数值；该临时图已清除。

### 完整测试命令

在 `E:\paper1\beamforming_demo`：

```powershell
python -m unittest tests.test_algorithm_audit evaluation.tests.test_evaluation_audit data.tests.test_data_audit
```

48 项通过，40.791 秒。

在 `E:\paper1\deeplearn`：

```powershell
python -m unittest tests.test_algorithm_audit evaluation.tests.test_evaluation_audit evaluation.tests.test_runtime_regressions data.scripts.tests.test_data_audit data.scripts.tests.test_download_epfl_interrupt
```

74 项通过，40.802 秒。新增问题先运行定向回归，再运行以上完整集合；第一轮根入口夹具缺少新必需属性，补齐夹具后通过，没有给生产代码加兜底。

### 关键路径冒烟与静态检查

- 每项目 7 个普通算法/模板分别做直接入口和 `python -m` 启动检查；其余 demo 12、deeplearn 19 个可执行文件逐个从所在目录执行 `python 文件名.py --help`，全部通过。公共库作为依赖检查。
- 两项目实际执行 `python run_one.py --scene simulation_contrast_speckle --algorithms das --dr 40 --no_tgc --no_evaluate --no_individual_images --output_root <临时目录>`，再增加 `--keep_existing` 重跑。
- 两次真实 H5 冒烟均验证 `(2,609,387)` 对比数组、有限值、算法峰值0 dB、GT逐值等于按H5生成范围解码的值、实际DR40/TGC关闭记录，以及第二次确实复用缓存。
- 修改的 Python：`python -m ruff check --select F,E9,I --fix <修改文件>` 通过；全部登记源码 AST 解析通过；两项目 `git diff --check` 通过。
- 两项目 17 个同路径 Python 文件逐字节一致；数据集专属实现独立保留。

## 验证边界与工作区

- 未执行实际 MBAN checkpoint 推理、全量仿真/EPFL 重打包、远端下载、断电/OS 强杀恢复。本轮不对这些路径做成功保证。
- 没有确认新的 P0/P1 留在已验证范围。未因理论异常添加清理、回退或兼容分支。
- 运行产物均在 `TemporaryDirectory` 内清除；已有结果与数据未删除或改写。
- demo 原有三张 `data/gt_preview` PNG 修改保留，并排除在本次提交之外。
