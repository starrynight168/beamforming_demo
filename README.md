# Beamforming Demo

一个简洁的超声波束合成对比工程。算法脚本可以单独运行，也可以通过 `run_one.py` / `run_all.py` 统一生成对比图和评估结果。

## 目录

```text
beamforming_demo/
  config.yaml
  check_data.py
  run_one.py
  run_all.py

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

## 评估说明

phantom 数据会计算：

- `contrast_dB`
- `speckle_pass_rate`
- `FWHM_axial_mm`
- `FWHM_lateral_mm`
- `distortion_pass_rate`

所有数据都会计算无参考辅助统计：

- `mean_raw_dB`
- `std_raw_dB`
- `mean_display_dB`
- `std_display_dB`
- `black_pixel_ratio`
- `white_pixel_ratio`

有 GT 的数据还会计算：

- `SSIM_vs_GT`
- `PSNR_dB_vs_GT`
- `MAE_dB_vs_GT`

in vivo 数据没有 GT 和 phantom，因此只生成图像对比和无参考统计。

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

只要遵守输入输出约定，`run_one.py` / `run_all.py` 会自动调用新算法、拼接对比图，并复用 `evaluation/evaluate.py` 生成指标。算法数量变多时，对比图会自动网格排版。
