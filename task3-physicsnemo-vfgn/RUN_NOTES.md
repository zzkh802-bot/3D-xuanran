# PhysicsNeMo / VFGN 本地跑通记录

## 文档对应模型

`D:\生产实习\端到端3D软件(1).docx` 中提到的 NVIDIA `PHYSICSNEMO` / `modulus`
对应 GitHub 仓库：

- https://github.com/NVIDIA/modulus
- 示例目录：`examples/additive_manufacturing/sintering_physics`
- 模型名：Virtual Foundary GraphNet / VFGN
- 论文：Virtual Foundry Graphnet for Metal Sintering Deformation Prediction

## 当前本机环境处理

本机默认 Python 是 3.9，而当前 PhysicsNeMo 主分支要求 Python 3.11+。
为了先跑通最小链路，做了本地兼容处理：

- sparse checkout 只拉取 `sintering_physics` 示例和必要 `physicsnemo` 源码目录。
- 给已拉取的 `physicsnemo` 源码加了 `from __future__ import annotations`。
- 用本地 `s3fs.py` stub 绕开不需要的 S3 导入。
- 给旧版 `warp-lang` 缺少的 `LOG_WARNING` 做了保护。
- 简化 `physicsnemo/models/__init__.py`，避免强制导入未 sparse checkout 的其他模型。

## 已跑通命令

从本目录运行：

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python run_synthetic_vfgn.py
python run_stl_vfgn_smoke.py
```

`run_synthetic_vfgn.py` 使用随机 8 节点图跑 VFGN 前向。

`run_stl_vfgn_smoke.py` 默认读取本任务目录内的样例 STL：

```text
sample_data\stl_foot.stl
```

采样 64 个顶点，构造 5 步模拟收缩序列，然后送入 VFGN。
输出保存到：

```text
outputs\stl_foot_vfgn_smoke_prediction.npy
```

## 重要限制

当前结果只是“模型计算链路跑通”，不是可信烧结预测。

原因：

- 官方仓库没有附带 README 中提到的预训练 checkpoint。
- 本地没有 VFGN 要求的真实烧结仿真 `tfrecord` 时间序列数据。
- 本地只有 STL 几何文件，所以脚本用 STL 顶点构造了模拟输入序列。

要做真实推理，需要补齐：

- 官方或训练得到的 VFGN checkpoint，例如 README 中的 `model_loss-*.pt`。
- 与模型 schema 匹配的 `metadata.json` 和 `*.tfrecord`。
- 或者完整物理仿真输出 `.pvtu/.vtu`，再通过 `data_process/rawdata2tfrecord_large_ts.py` 转换。
