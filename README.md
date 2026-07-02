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
    fdmas.py

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

1. 将新算法脚本放入 `algorithms/`，例如 `algorithms/new_method.py`
2. 推荐先复制模板：

```bash
cp algorithms/template_algorithm.py algorithms/new_method.py
```

然后修改：

```python
METHOD_NAME = "new_method"
```

并实现模板中的：

```python
def beamform(data, args):
    ...
    return image_db
```

3. 算法脚本至少支持这些参数：

```bash
python algorithms/new_method.py \
  --h5_path data/simulation.h5 \
  --h5_sample_idx 0 \
  --output_dir results/test \
  --dr 60
```

4. 输出目录必须使用算法名作为子文件夹：

```text
output_dir/new_method/new_method.npy
output_dir/new_method/new_method.png
output_dir/new_method/params.json
```

其中 `new_method.npy` 是 dB 图像，形状必须和 `x_grid/z_grid` 对应的成像网格一致。

5. 临时对比时不需要改配置，直接运行：

```bash
python run_one.py --scene simulation_contrast_speckle --algorithms das,mv,new_method
```

也可以跑全部场景：

```bash
python run_all.py --algorithms das,mv,new_method
```

6. 如果希望它成为默认算法，再在 `config.yaml` 中加入：

```yaml
algorithms:
  - das
  - mv
  - new_method
```

只要遵守以上输入输出约定，`run_one.py` / `run_all.py` 会自动调用新算法、拼接对比图，并复用 `evaluation/evaluate.py` 生成指标。
