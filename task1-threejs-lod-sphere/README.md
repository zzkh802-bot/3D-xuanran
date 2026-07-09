# 任务一：Three.js 四级 LOD 球面知识图谱

## 运行

```powershell
npm install
npm run dev
```

打开本地地址：`http://127.0.0.1:5173`

也可以直接聚焦某一门课，例如：

```text
http://127.0.0.1:5173/?course=gaoshu-xia
```

生产构建：

```powershell
npm run build
```

## 已包含功能

- 四个教材球：高等数学上、高等数学下、线性代数、概率论与数理统计。
- 总览 LOD0：只显示球体和外层端点。
- 点击教材球或右侧按钮后聚焦单球。
- 拉近/拉远自动切换四级 LOD：章节区域、小节区域、知识点。
- 分区为贴合球面的细分曲面，不再只是平面直线拼接。
- 章节/小节文字以纹理方式贴近球面表面。
- OrbitControls 支持拖拽旋转、缩放。
- 微调模式支持选中点后改 `phi/psi`、方向按钮微调，也支持在球面上拖动点。

## 数据

- `src/data/course-data.json`：前端运行数据。
- `scripts/generate_course_data.py`：从四个课程变量名脚本重新生成 JSON。
- `src/data/*.py` 和 `src/data/*.xlsx`：保留的原始课程数据。

重新生成课程 JSON：

```powershell
python scripts\generate_course_data.py
```

本次生成结果：

- 高等数学上：7 章，40 小节。
- 高等数学下：5 章，33 小节。
- 线性代数：6 章，28 个区域。
- 概率论与数理统计：8 章，30 小节。

## 验证

- `npm run build` 已通过。
- 精简交付目录不保留 `node_modules`、`dist` 和截图；需要时按上面的命令重新安装、构建、运行。
