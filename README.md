# Beamforming Demo

一个简洁的超声波束合成对比工程。算法脚本可以单独运行，也可以通过 `run_one.py` / `run_all.py` 统一生成对比图和评估结果。

## 目录

```text
beamforming_demo/
  config.yaml
  check_data.py
  run_one.py
  run_all.py
  run_wizard_cn.py
  run_ablation_cn.py

  data/
    pack_data.py
    simulation.h5
    experiments.h5
    in_vivo.h5

  algorithms/
    das.py
    mv.py
    esbmv.py
    gcfmv.py
    cmsaw.py
    fdmas.py
    template_algorithm.py

  evaluation/
    evaluate.py
    plot_metrics.py

  results/
```

## 环境

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate beamforming-demo
```

也可以在已有 Python 环境中安装依赖：

```bash
pip install -r requirements.txt
```

默认配置使用 CUDA 12.8 版 PyTorch。若机器没有 NVIDIA GPU，请按 PyTorch 官网说明安装 CPU 版 `torch`。

## 准备数据

**重要**：原始数据 `PICMUS` 文件夹因大小限制未包含在代码仓库中，请从以下链接下载压缩包：
- `PICMUS` 数据压缩包：https://drive.google.com/file/d/1CQxjvpwGHDyzwHSJQux-mkXLl97mul-f/view?usp=drive_link

下载后，将压缩包解压，并用其中的 `PICMUS` 文件夹**替换**项目根目录下的 `PICMUS/` 文件夹（如果已存在）。确保解压后的路径结构（如 `PICMUS/simulation/...`）与 `data/pack_data.py` 脚本中的预期路径一致。

如果 `data/*.h5` 已存在，可以直接运行。需要重新从 `PICMUS/` 打包算法输入数据时：

```bash
python data/pack_data.py
```

检查数据是否完整：

```bash
python check_data.py
```

打包后的数据：

- `data/simulation.h5`: simulation contrast/speckle 和 resolution/distortion
- `data/experiments.h5`: experiments contrast/speckle 和 resolution/distortion
- `data/in_vivo.h5`: carotid cross 和 carotid long

**备用下载**：如果上述链接访问不便，也可以从以下项目文件夹链接获取完整工程（包含数据）：
https://drive.google.com/drive/folders/1HeaowzynJdmK188EPwCqfgtj5zPGieH8?usp=drive_link

## 单独运行一个算法

```bash
python algorithms/das.py --h5_path data/simulation.h5 --h5_sample_idx 0 --output_dir results/simulation_contrast_speckle
```

输出：

```text
results/simulation_contrast_speckle/
  das/
    das.npy
    das.png
    params.json
```

单独运行算法只生成该算法自己的图和 `.npy`，不做多算法对比和评估。

## 运行一个场景

```bash
python run_one.py
```

输出：

```text
results/simulation_contrast_speckle/
  das/
  mv/
  esbmv/
  gcfmv/
  cmsaw/
  fdmas/
  comparison.npy
  comparison.png
  run_params.json
  metrics/
```

`metrics/` 中包含：

- `summary_metrics.csv`
- `standard_metrics.png`
- `auxiliary_metrics.png`
- `roi_targets.png`（phantom 数据）
- `contrast_roi_metrics.csv`
- `resolution_target_metrics.csv`
- `contrast_group_metrics.csv`
- `resolution_group_metrics.csv`
- `contrast_group_metrics.png`
- `resolution_group_metrics.png`
- `picmus_challenge_summary.txt`
- `evaluation_meta.json`

`run_one.py` 会先调用 `evaluation/evaluate.py` 计算指标并写出 CSV / JSON / TXT，再调用 `evaluation/plot_metrics.py` 从这些表格生成图片。`run_all.py` 通过逐个调用 `run_one.py` 复用同一流程。

## 一键运行全部场景

```bash
python run_all.py
```

只运行部分场景：

```bash
python run_all.py --only simulation_contrast_speckle,carotid_cross
```

只运行部分算法：

```bash
python run_one.py --scene carotid_cross --algorithms das,mv
```

## 中文交互向导

项目提供了两个面向中文用户的交互式脚本，用于辅助参数配置和单参数消融实验。

### 1. 成像向导：`run_wizard_cn.py`

适合对命令行参数不熟悉的“小白”用户。该脚本会以全中文一步步引导您配置超声成像的各项参数：

- **引导式输入**：按顺序选择 H5 数据集、样本索引、波束合成算法、发射偏角、显示参数（如动态范围 `dr`）、TGC（时间增益补偿）、窗函数、插值方式、是否保存 GT 及进行指标评估等。
- **人性化操作**：在任何步骤输入 `b` 可返回上一步重新选择，输入 `q` 可直接退出。
- **小白说明模式**：每一步提供详细的背景知识说明，解释该参数对最终成像画质的具体影响。
- **智能分支逻辑**：例如关闭“动态孔径”后自动跳过 `F-Number` 的提问；关闭 “TGC” 后跳过 `TGC_ALPHA` 提问。
- **参数容错与校验**：输入的参数格式不正确时会给出中文报错并提示重新输入，不会直接崩溃。例如自定义角度必须为 `center`、`all`、正整数或整数索引列表。
- **输出路径**：
  - 默认结果输出在 `results/wizard/` 目录下。
  - 临时取消执行不会写入配置，执行时会将当前的配置文件保存到 `results/wizard/configs/`。

使用方法：

```bash
python run_wizard_cn.py
```

---

### 2. 单参数消融向导：`run_ablation_cn.py`

用于对某一个特定参数进行“消融实验”（Ablation Study），观察其取值变化对算法成像质量的影响。

- **多参数列表展示**：固定一个算法后，自动从 `config.yaml` 读取所有可消融的超参数，并以中文表格列出。
- **单变量控制**：一次只允许对一个选定参数进行消融实验。
- **灵活输入取值**：支持手动列举多个取值（逗号分隔），或通过范围生成（如 `1.2:0.1:2.0` 表示从 1.2 到 2.0，步长 0.1）。
- **参数强类型校验**：例如 `select_angles` 参数不接受小数或布尔值，数值参数只能输入数字，开关参数只接受 true/false，`window` 和 `interp` 只接受合法选项。
- **输出结构设计**：
  默认输出到 `results/ablation/ablation_<算法名>_<参数名>/`，如 `ablation_das_f_number/`。结构如下：
  ```text
  results/ablation/ablation_das_f_number/
    configs/                      # 每次运行生成的临时配置文件
    fnumber1.2/                   # 对应参数值的原生结果目录
      comparison.png
      comparison.npy
      run_params.json
      das/
        das.npy
    fnumber1.5/
    ...
    ablation_summary.csv          # 指标消融汇总 CSV
    ablation_comparison.png       # 自动生成的对比拼接总览图
    ablation_comparison.npy       # 对比图像数据
  ```
- **智能对比总览图**：消融结束后，脚本会自动读取每个子目录中的 `.npy` 图像数据，将 Ground Truth（若有）作为第一张，随后拼接各个参数取值的成像图，生成 `ablation_comparison.png`。子图布局支持自动折行（每行最多 4 张），并优先读取自身配置的 `dr` 动态范围进行完美显示。

使用方法：

```bash
python run_ablation_cn.py
```

## 评估说明

评估和绘图已经拆分：

```text
evaluation/evaluate.py      # 只计算指标，写 CSV / JSON / TXT
evaluation/plot_metrics.py  # 只读取评估输出并画 PNG
```

单独重新计算指标：

```bash
python evaluation/evaluate.py \
  --comparison_npy results/simulation_contrast_speckle/comparison.npy \
  --h5_path data/simulation.h5 \
  --h5_sample_idx 0 \
  --out_dir results/simulation_contrast_speckle/metrics
```

单独重新画图：

```bash
python evaluation/plot_metrics.py \
  --metrics_dir results/simulation_contrast_speckle/metrics
```

phantom 数据会计算主指标：

- `contrast_dB`
- `CR_dB`
- `CNR`
- `gCNR`
- `cyst_residual_dB`
- `speckle_pass_rate`
- `speckle_KS_D`
- `speckle_KS_p`
- `speckle_SNR`
- `ENL`
- `FWHM_axial_mm`
- `FWHM_lateral_mm`
- `PSLR_dB`
- `ISLR_dB`
- `distortion_mm`
- `distortion_pass_rate`

有 GT 的数据还会计算：

- `SSIM_vs_GT`
- `PSNR_dB_vs_GT`
- `MAE_dB_vs_GT`

PICMUS 分组结果会写入：

- `contrast_group_metrics.csv`
- `resolution_group_metrics.csv`
- `picmus_challenge_summary.txt`

对应分组图由 `plot_metrics.py` 生成：

- `contrast_group_metrics.png`
- `resolution_group_metrics.png`

in vivo 数据使用多角度 DAS 生成的 reference 作为 GT，因此会计算 `SSIM_vs_GT`、`PSNR_dB_vs_GT`、`MAE_dB_vs_GT` 等参考指标；由于没有 phantom ROI/target，contrast/resolution 分组指标为空。

`plot_metrics.py` 会根据算法数量自适应图像布局。算法较多时，普通指标图会自动改为横向柱状图；指标太多时会按指标分页，例如 `standard_metrics_page2.png`。

## 添加新算法

推荐方式是从模板复制一个新脚本。只要脚本名、`METHOD_NAME`、输出文件名一致，`run_one.py` / `run_all.py` 就可以通过 `config.yaml` 自动调用它。

### 1. 复制模板

```bash
cp algorithms/template_algorithm.py algorithms/new_method.py
```

然后在 `algorithms/new_method.py` 中修改：

```python
METHOD_NAME = "new_method"
```

注意三者必须一致：

```text
脚本名:      algorithms/new_method.py
METHOD_NAME: new_method
输出文件:    output_dir/new_method/new_method.npy
```

### 2. 实现算法主体

实现模板中的：

```python
def beamform(data, args):
    ...
    return image_db
```

`beamform()` 输入里的主要字段：

```python
data["I"]              # [angles, time, channels]
data["Q"]              # [angles, time, channels]
data["t0"]
data["fs"]
data["c"]
data["fc"]
data["pitch"]
data["num_channels"]
data["z_grid"]
data["x_grid"]
data["angles"]
data["selected_angles"]
data["selected_angle_indices"]
```

返回值必须是二维 dB 图像：

```text
shape = [len(z_grid), len(x_grid)]
最大值通常归一化到 0 dB
```

### 3. 保留通用命令行参数

模板已经包含 `run_one.py` 会传入的通用参数。新算法即使用不到，也建议保留：

```text
--h5_path
--h5_sample_idx
--output_dir
--select_angles
--f_number
--dr
--dynamic_aperture / --no_dynamic_aperture
--tgc / --no_tgc
--tgc_alpha
--window
--interp
--save_gt
```

可以单独测试：

```bash
python algorithms/new_method.py \
  --h5_path data/simulation.h5 \
  --h5_sample_idx 0 \
  --output_dir results/test \
  --dr 60
```

输出目录必须使用算法名作为子文件夹：

```text
output_dir/new_method/new_method.npy
output_dir/new_method/new_method.png
output_dir/new_method/params.json
```

### 4. 加入 `config.yaml`

如果希望默认参与 `run_one.py` / `run_all.py`，在 `config.yaml` 顶部加入算法名，并在 `algorithm_params` 里写该算法的默认超参数：

```yaml
algorithms:
  - das
  - mv
  - esbmv
  - gcfmv
  - cmsaw
  - fdmas
  - new_method

algorithm_labels:
  new_method: New Method

algorithm_params:
  new_method:
    alpha: 0.5
    num_iter: 10
    use_filter: true
```

之后直接运行：

```bash
python run_one.py --scene simulation_contrast_speckle
```

或运行全部场景：

```bash
python run_all.py
```

临时测试时也可以不改配置：

```bash
python run_one.py --scene simulation_contrast_speckle --algorithms das,mv,new_method
```

### 5. 方法专属参数

如果新算法有自己的参数，例如：

```bash
--alpha 0.5
--num_iter 10
```

优先在算法脚本中设置默认值：

```python
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--num_iter", type=int, default=10)
```

统一实验时，把默认超参数写在 `config.yaml` 的 `algorithm_params` 中：

```yaml
algorithm_params:
  new_method:
    alpha: 0.8
    num_iter: 20
```

`run_one.py` / `run_all.py` 会自动把它转换成：

```text
--alpha 0.8 --num_iter 20
```

布尔参数会自动转换。`true` 会变成 `--use_filter`，`false` 会变成 `--no_use_filter`。因此如果算法支持从配置关闭某个布尔参数，脚本里要同时提供正反两个参数：

```python
parser.add_argument("--use_filter", action="store_true", default=True)
parser.add_argument("--no_use_filter", dest="use_filter", action="store_false")
```

普通新算法不需要修改 `run_one.py`。只要脚本支持 argparse，并且 `config.yaml` 里算法名和 `algorithm_params` 块名称一致即可。只有特别复杂的调度规则，例如某个参数要根据场景自动变化、或者一个参数需要转换成多个命令行参数时，才需要改 `run_one.py`。

### 6. 显示名称可选

如果不改显示名称，图中会显示算法名的大写形式。想要更漂亮的名称，在 `config.yaml` 里加入：

```yaml
algorithm_labels:
  new_method: New Method
```

这不是必需步骤，不影响运行。

只要遵守输入输出约定，`run_one.py` / `run_all.py` 会自动调用新算法、拼接对比图，并复用 `evaluation/evaluate.py` 生成指标、`evaluation/plot_metrics.py` 生成评估图。算法数量变多时，对比图和评估图会自动调整布局。

