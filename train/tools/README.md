# 训练审计工具

这里的工具只读取 checkpoint 或日志，不修改训练配置、checkpoint 或硬件参数。

- `qat_audit.py`：核对 Bias 的有效量化值、阵列共享量程、普通权重量化步长和量化误差。
- `training_log.py`：提取逐轮 data loss、Lrange、数值健康和映射尺度。

模型推理、硬件 trace、mapping 和 PPA 继续使用 `evaluate_mban.py`，避免在工具目录复制第二套推理实现。
