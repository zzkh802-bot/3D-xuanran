# 任务一：Three.js 四级 LOD 球面知识图谱

## 运行

不要直接双击 `index.html`；这个页面使用 Three.js ES module 和 JSON import，需要通过本地 Vite 服务访问。

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
- 拉近/拉远自动切换四级 LOD：教材、章节、小节、知识点。
- 每门课只渲染一个不透明球面；章节色、小节色和凹槽由 Shader 内的语义场控制，不再叠加区域网格。
- 章、节使用独立边界距离场；小节层级仍保留更宽、更深、更暗的章节边界。
- 章节/小节文字采用统一目标字号，仅在极小区域内有限缩放，并以局部弯曲网格贴近球面。
- 小节区域严格限制在所属章节内；凹区域使用有向球面面积判定内外侧，重叠和空隙按最近区域中心稳定归属。
- OrbitControls 支持拖拽旋转、缩放。
- 微调模式支持选中共享交点后改 `phi/psi`、方向按钮微调，也支持按住 Shift 在球面上拖动；关联边界同步变化，松手后重建语义场。
- 未被任何章/节边界引用的辅助点不会显示为外层点或编辑控制点。
- LOD3 保留小节名并显示知识点，同时隐藏外层代表点。

## 数据

- `src/data/course-data.json`：前端直接加载的 schema v3 数据。
- `scripts/generate_course_data.py`：读取 XLSX 中的节路径、球面坐标、点对和章范围，生成共享顶点 ID 数据。
- `src/data/*.xlsx`：章、节和交点坐标的主要数据源。
- `src/data/*.py`：保留的参考曲线脚本；目前仅用于补齐线性代数表格缺失的 `P34`。

重新生成课程 JSON：

```powershell
python -m pip install -r requirements.txt
python scripts\generate_course_data.py
```

本次生成结果：

- 高等数学上：7 章，40 小节。
- 高等数学下：5 章，33 小节。
- 线性代数：6 章，27 个小节区域。
- 概率论与数理统计：8 章，30 小节。

## 验证

- `python scripts\generate_course_data.py` 已通过，全部章/节引用均能解析到共享顶点。
- `npm run build` 已通过。
- Playwright 已验证四本教材 LOD1、高等数学下 LOD1/2/3、390x844 移动端和 Shift 拖拽联动。
- 精简交付目录不保留 `node_modules`、`dist` 和截图；需要时按上面的命令重新安装、构建、运行。
