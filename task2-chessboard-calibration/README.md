# 任务二：棋盘位图识别与标定

## 主脚本

- `yanzheng2026_5_12.py`

## 依赖数据

- `实习资料/2022nov1111`：精简样例位图，保留 `hewei025.bmp` 到 `hewei031.bmp`。
- `实习资料/saddle_and_mark_from_m.py`：鞍点检测、角点精化、标记点逻辑。

全量位图和大体积 MATLAB 参考包未放入精简交付目录；需要全量测试时，把完整图片目录传给 `--img-folder`。

## 运行

单张或小批量：

```powershell
$env:MPLBACKEND='Agg'
python yanzheng2026_5_12.py --img-folder "实习资料\2022nov1111" --start 3 --limit 1 --save-dir "识别结果\恢复验证" --no-show
```

使用外部全量图片测试前 75 张：

```powershell
$env:MPLBACKEND='Agg'
python yanzheng2026_5_12.py --img-folder "D:\生产实习\生产实习\实习资料\2022nov1111" --start 0 --limit 75 --save-dir "识别结果\前75测试" --no-show
```

全量覆盖输出时，把 `--limit` 去掉并把 `--save-dir` 指到目标输出目录。

## 当前恢复/完善点

- 结合 BFS 生长和旧单应矩阵生长生成候选棋盘。
- 用行列残差评分抑制偏移、斜线候选。
- 对候选点做最终行列离群点剔除。
- 对跳格修正加门控，避免错误加密导致漂移。
- 单应网格补全一圈后做稀疏边界裁剪。
- 白色十字绘制在最上层，红线为竖向边，蓝线为横向边。
- CLI 支持 `--img-folder`、`--start`、`--limit`、`--save-dir`、`--no-show`。

## 验证

- `python -m py_compile yanzheng2026_5_12.py` 已通过。
- `hewei028.bmp` 单张 no-show 验证已通过，重建 160 个交叉点。
- 精简交付目录不保留运行输出；需要时按上面的命令重新生成 `识别结果`。
