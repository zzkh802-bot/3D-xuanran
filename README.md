# 3D xuanran

本目录已按两个任务拆分完成。

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

## 任务二：棋盘位图识别与标定

目录：`task2-chessboard-calibration`

内容包括恢复后的主脚本、支持脚本和 7 张可跑通的样例位图。全量位图和大体积 MATLAB 参考包未放入精简交付目录。

单张验证：

```powershell
cd "D:\生产实习\3D xuanran\task2-chessboard-calibration"
$env:MPLBACKEND='Agg'
python yanzheng2026_5_12.py --img-folder "实习资料\2022nov1111" --start 3 --limit 1 --save-dir "识别结果\恢复验证" --no-show
```
