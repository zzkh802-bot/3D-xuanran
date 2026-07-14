# 3D xuanran

本目录已按三个任务拆分完成。

## 任务一：Three.js 四级 LOD 球面知识图谱

目录：`task1-threejs-lod-sphere`

内容包括 Three.js/Vite 前端、四门教材源数据、课程 JSON 生成脚本。依赖和构建产物未放入精简交付目录。

运行：

```powershell
cd "D:\生产实习\3D xuanran\task1-threejs-lod-sphere"
npm install
npm run dev
```

运行后打开：`http://127.0.0.1:5173`

不要直接双击 `index.html`，需要通过上面的 Vite 本地服务访问。

## 任务二：棋盘位图识别与标定

目录：`task2-chessboard-calibration`

内容包括恢复后的主脚本、支持脚本和 7 张可跑通的样例位图。全量位图和大体积 MATLAB 参考包未放入精简交付目录。

单张验证：

```powershell
cd "D:\生产实习\3D xuanran\task2-chessboard-calibration"
$env:MPLBACKEND='Agg'
python yanzheng2026_5_12.py --img-folder "实习资料\2022nov1111" --start 3 --limit 1 --save-dir "识别结果\恢复验证" --no-show
```

## 任务三：PhysicsNeMo / VFGN 烧结变形模型 smoke test

目录：`task3-physicsnemo-vfgn`

内容包括 NVIDIA Modulus/PhysicsNeMo 中 `sintering_physics` 示例的必要源码、本机 Python 3.9 兼容补丁、两个可运行 smoke test 脚本和一个本地 STL 样例。

运行：

```powershell
cd "D:\生产实习\3D xuanran\task3-physicsnemo-vfgn"
$env:PYTHONPATH=(Resolve-Path .).Path
python run_synthetic_vfgn.py
python run_stl_vfgn_smoke.py
```

说明：当前 smoke test 用随机小数据和 `sample_data\stl_foot.stl` 构造输入，验证 GitHub 模型链路能跑通；没有官方 checkpoint 和真实 `tfrecord`，所以不是可信烧结预测结果。
