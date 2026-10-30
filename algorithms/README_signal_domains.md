# 信号域入口

训练在 `train/config.yaml` 的 `target.signal_domain` 选择 iq、analytic_rf、rf，输入和教师必须同域。教师必须事先保存；输入优先读取所选域通道缓存，缺少缓存才从 RF 在内存派生，转换不会写入数据集。解析 RF 通道缓存为 ar_i/ar_q，和 MV 教师 mv_ar_i/mv_ar_q 不同。RF 模式有效监督组使用 objective:iq（比较实部/虚部，教师虚部为零）；不允许 envelope 冒充 RF 包络。详见 `train/docs/signal_domains.md`。

算法命令统一支持：

```powershell
python -m algorithms.mv --h5_path data/volunteer_005.h5 --signal_domain analytic_rf --select_angles 1 --dynamic_aperture --output_dir results/analytic
python -m algorithms.das --h5_path data/volunteer_005.h5 --signal_domain rf --output_dir results/rf
```

DAS、MV、CC-MV、ESBMV、GCF-MV、CMSAW 支持三域。MV 系列在三域均支持 rx/tx/joint，tx/joint 要求至少两个不同发射角。F-DMAS 的实 RF 路径在密深度网格上进行带符号成对运算，再用 Kaiser FIR 提取 `[1.5fc, 2.5fc]` 的二次谐波带；IQ 和解析 RF 使用复数信号路径，其中 IQ 是复基带变体。IQ 专用相位补偿只在 IQ 开启，不把 RF 下变频后称为 RF 成像。

RF 命令行输出 B-mode 时临时密深度计算、深端补 N 个零后 Hilbert 取包络、回到原图像网格；参数文件记录输入域和计算行数。解析RF/IQ直接取复数幅值。算法GT按所选域读取，不借另一域显示缓存。

MBAN默认使用checkpoint域；显式 --signal_domain 必须与checkpoint一致。RF训练预览是有符号图；评估 reconstruct_signal 返回原网格复数输出，reconstruct 返回B-mode，RF路径临时密采样并记录预测虚部RMS。

根目录比较入口可在config.yaml的params.signal_domain选择域。默认iq。混域checkpoint应分别评估，避免同一次运行共享错误采样/相位状态。现有数据没有重新生成。

## 本次验证

- `python train/run_tests.py --module all`：运行 587 项，发现两处旧点靶夹具未声明主算法；补齐后两项逐项复验通过，其余 585 项通过。
- 根目录原 94 项回归覆盖算法、缓存和旧 IQ 行为；新增两项测试逐个实跑 RF-only H5 的七个算法/三个域组合，并验证 RF 密网格显示和不支持组合报错。旧夹具新增域声明后复验通过。
- `python -m unittest discover -s data/scripts/tests -q`：141 项通过。
- 三域训练专项覆盖 explicit/factorized、缓存/非缓存、梯度和实际参数更新、checkpoint 跨域拒绝。临时数据均自动清理；没有重打包生产数据或训练完整模型。
