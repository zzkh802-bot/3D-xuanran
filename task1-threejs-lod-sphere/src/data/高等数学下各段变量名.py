import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===================== 1. 全局基础参数 =====================
R = 10.0        # 球面半径
r1 = 6.0        # z=h1平面圆弧半径
h1 = 3.0        # 上圆弧平面高度 z=h1
h2 = 6.0        # 下点位平面高度 z=-h2
sample_num = 80 # 曲线采样点数（控制平滑度）

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 下平面径向长度 sqrt(R² - h2²)
rad_h2 = np.sqrt(R ** 2 - h2 ** 2)

# ===================== 2. 通用工具函数（文档标准公式） =====================
def point_to_sphere(pt):
    """单点映射到半径R的球面（文档给定公式）"""
    x, y, z = pt
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0])
    return np.array([R*x/norm, R*y/norm, R*z/norm])

def curve_to_sphere(curve_pts):
    """整条曲线批量映射到球面"""
    return np.array([point_to_sphere(p) for p in curve_pts])

def get_arc_curve(t_start, t_end):
    """生成z=h1平面圆弧原始点（文档标准参数方程）"""
    t_arr = np.linspace(t_start, t_end, sample_num)
    x = r1 * np.cos(t_arr)
    y = r1 * np.sin(t_arr)
    z = np.full_like(t_arr, h1)
    return np.column_stack([x, y, z])

def get_segment_curve(p1, p2):
    """生成两点间线段原始点（文档参数方程）"""
    t_arr = np.linspace(0, 1, sample_num)
    return np.array([(1-t)*np.array(p1) + t*np.array(p2) for t in t_arr])

def mid_point(p1, p2):
    """两点中点（文档定义）"""
    return 0.5 * (np.array(p1) + np.array(p2))

def five_equal_points(p1, p2):
    """两点五等分，返回中间4个点（文档定义）"""
    return [(1-k/5)*np.array(p1) + (k/5)*np.array(p2) for k in range(1, 5)]

def three_div_points(p1, p2):
    """两点三等分：(近p1分点, 近p2分点)（文档定义）"""
    pt1 = (2/3)*np.array(p1) + (1/3)*np.array(p2)
    pt2 = (1/3)*np.array(p1) + (2/3)*np.array(p2)
    return pt1, pt2

def build_closed_curve(seg_list):
    """拼接多段曲线 + 整体映射到球面，返回完整封闭曲线"""
    full_curve = np.vstack(seg_list)
    return curve_to_sphere(full_curve)

# ===================== 新增：仅收集【线段/圆弧真正端点】，剔除纯中间点 =====================
def collect_all_label_points():
    """
    只保留：圆弧端点、线段端点
    剔除：P35 等非端点中间点
    """
    # 人工筛选：所有出现在圆弧/线段首尾的变量
    base_points = {
        # 上层圆弧主端点
        'A': A, 'B': B, 'tp': tp, 'C': C,
        'P11': P11, 'P12': P12, 'P13': P13, 'P14': P14,
        'P15': P15, 'P16': P16, 'P17': P17, 'P18': P18,
        'O1': O1,

        # 下层基准顶点（线段端点）
        'D': D, 'E': E, 'Df': Df, 'Ef': Ef, 'F': F,

        # 线段连接用端点（曲线构成顶点）
        'G': G, 'H': H, 'I': I,
        'P21': P21, 'P22': P22, 'P23': P23, 'P24': P24,
        'P25': P25,
        'P26': P26, 'P27': P27,

        'P31': P31, 'P32': P32, 'P33': P33, 'P34': P34,
        'P36': P36, 'P37': P37, 'P38': P38, 'P39': P39,
        'P310': P310, 'P311': P311, 'P312': P312
    }

    # 坐标去重，避免浮点误差重复标注
    unique_points = {}
    seen_coords = set()
    for name, coord in base_points.items():
        coord_tuple = tuple(round(x, 4) for x in coord)
        if coord_tuple not in seen_coords:
            seen_coords.add(coord_tuple)
            unique_points[name] = coord
    return unique_points

# ===================== 3. 所有圆弧角度参数（文档原文区间） =====================
t_A      = -5 * np.pi / 6
t_B      = -np.pi / 6
t_tp     = -np.pi / 2
t_C      = np.pi / 2
t_A_end  = 7 * np.pi / 6  # 与-5π/6同角，用于闭合圆弧

# P11~P18对应角度（文档圆弧区间端点）
t_P11    = t_A + 2*np.pi/15
t_P12    = t_A + 4*np.pi/15
t_P13    = t_A + 6*np.pi/15
t_P14    = t_A + 8*np.pi/15
t_P15    = np.pi / 6
t_P16    = t_C + 2*np.pi/9
t_P17    = 5*np.pi / 6
t_P18    = t_C + 4*np.pi/9

# ===================== 4. 全部基础关键点（严格按文档定义） =====================
# 4.1 z=h1平面圆弧端点（P1i系列）
def get_arc_point(t):
    """根据角度获取z=h1平面上的点"""
    return np.array([r1*np.cos(t), r1*np.sin(t), h1])

A    = get_arc_point(t_A)
B    = get_arc_point(t_B)
tp   = get_arc_point(t_tp)
C    = get_arc_point(t_C)
P11  = get_arc_point(t_P11)
P12  = get_arc_point(t_P12)
P13  = get_arc_point(t_P13)
P14  = get_arc_point(t_P14)
P15  = get_arc_point(t_P15)
P16  = get_arc_point(t_P16)
P17  = get_arc_point(t_P17)
P18  = get_arc_point(t_P18)

O1   = np.array([0.0, 0.0, h1])  # 上平面圆心

# 4.2 z=-h2平面定点（文档原始坐标）
D  = np.array([-np.sqrt(3)/2 * rad_h2,  -1/2 * rad_h2, -h2])
E  = np.array([ np.sqrt(3)/2 * rad_h2,  -1/2 * rad_h2, -h2])
Df = np.array([-3*np.sqrt(3)/10 * rad_h2, 1/10 * rad_h2, -h2])
Ef = np.array([ 3*np.sqrt(3)/10 * rad_h2, 1/10 * rad_h2, -h2])
F  = np.array([0.0, rad_h2, -h2])

# 4.3 中点（修正所有OCR笔误）
G    = mid_point(A, D)
H    = mid_point(B, E)
I    = mid_point(C, F)
P25  = mid_point(H, I)
P35  = mid_point(D, E)       # 已从标注列表移除（非端点）
P36  = mid_point(E, F)
P38  = mid_point(F, D)
P312 = mid_point(P36, P38)

# 4.4 五等分点
P21, P22, P23, P24 = five_equal_points(G, H)
P31, P32, P33, P34 = five_equal_points(D, E)

# 4.5 三等分点
P37, P39 = three_div_points(F, D)
P310, P311 = three_div_points(P38, P36)

# 4.6 推导点位
P26, P27 = three_div_points(I, G)

# ===================== 5. 所有圆弧段预生成（文档原文区间） =====================
arc = {
    "Atp":      get_arc_curve(t_A, t_tp),
    "tpB":      get_arc_curve(t_tp, t_B),
    "BP15":     get_arc_curve(t_B, t_P15),
    "P15C":     get_arc_curve(t_P15, t_C),
    "CP17":     get_arc_curve(t_C, t_P17),
    "P17A":     get_arc_curve(t_P17, t_A_end),
    "AP11":     get_arc_curve(t_A, t_P11),
    "P11P12":   get_arc_curve(t_P11, t_P12),
    "P12P13":   get_arc_curve(t_P12, t_P13),
    "P13P14":   get_arc_curve(t_P13, t_P14),
    "P14B":     get_arc_curve(t_P14, t_B),
    "CP16":     get_arc_curve(t_C, t_P16),
    "P16P18":   get_arc_curve(t_P16, t_P18),
    "P18A":     get_arc_curve(t_P18, t_A_end),
}

# ===================== 6. 逐章节构建封闭曲线（100%对齐文档原文） =====================
chapter_curves = {}

# -------- 第八章 空间解析几何（6节） --------
chapter_curves["8-1 向量及其线性运算"] = build_closed_curve([
    arc["Atp"], get_segment_curve(tp, O1), get_segment_curve(O1, A)
])
chapter_curves["8-2 数量积·向量积·混合积"] = build_closed_curve([
    arc["tpB"], get_segment_curve(B, O1), get_segment_curve(O1, tp)
])
chapter_curves["8-3 曲面及其方程"] = build_closed_curve([
    arc["BP15"], get_segment_curve(P15, O1), get_segment_curve(O1, B)
])
chapter_curves["8-4 空间曲线及其方程"] = build_closed_curve([
    arc["P15C"], get_segment_curve(C, O1), get_segment_curve(O1, P15)
])
chapter_curves["8-5 平面及其方程"] = build_closed_curve([
    arc["CP17"], get_segment_curve(P17, O1), get_segment_curve(O1, C)
])
chapter_curves["8-6 空间直线及其方程"] = build_closed_curve([
    arc["P17A"], get_segment_curve(A, O1), get_segment_curve(O1, P17)
])

# -------- 第九章 多元函数微分法及其应用（10节） --------
chapter_curves["9-1 多元函数的基本概念"] = build_closed_curve([
    arc["AP11"], get_segment_curve(P11, P21), get_segment_curve(P21, G), get_segment_curve(G, A)
])
chapter_curves["9-2 偏导数"] = build_closed_curve([
    arc["P11P12"], get_segment_curve(P12, P22), get_segment_curve(P22, P21), get_segment_curve(P21, P11)
])
chapter_curves["9-3 全微分"] = build_closed_curve([
    arc["P12P13"], get_segment_curve(P13, P23), get_segment_curve(P23, P22), get_segment_curve(P22, P12)
])
chapter_curves["9-4 多元复合函数的求导法则"] = build_closed_curve([
    arc["P13P14"], get_segment_curve(P14, P24), get_segment_curve(P24, P23), get_segment_curve(P23, P13)
])
chapter_curves["9-5 隐函数的求导公式"] = build_closed_curve([
    arc["P14B"], get_segment_curve(B, H), get_segment_curve(H, P24), get_segment_curve(P24, P14)
])
chapter_curves["9-6 多元函数微分学的几何应用"] = build_closed_curve([
    get_segment_curve(P24, H), get_segment_curve(H, E),
    get_segment_curve(E, P34), get_segment_curve(P34, P24)
])
chapter_curves["9-7 方向导数与梯度"] = build_closed_curve([
    get_segment_curve(P23, P24), get_segment_curve(P24, P34),
    get_segment_curve(P34, P33), get_segment_curve(P33, P23)
])
chapter_curves["9-8 多元函数的极值及其求法"] = build_closed_curve([
    get_segment_curve(P22, P23), get_segment_curve(P23, P33),
    get_segment_curve(P33, P32), get_segment_curve(P32, P22)
])
chapter_curves["9-9 二元泰勒公式"] = build_closed_curve([
    get_segment_curve(P21, P22), get_segment_curve(P22, P32),
    get_segment_curve(P32, P31), get_segment_curve(P31, P21)
])
chapter_curves["9-10 最小二乘法"] = build_closed_curve([
    get_segment_curve(G, P21), get_segment_curve(P21, P31),
    get_segment_curve(P31, D), get_segment_curve(D, G)
])

# -------- 第十章 重积分（4节） --------
chapter_curves["10-1 二重积分的概念与性质"] = build_closed_curve([
    arc["BP15"], get_segment_curve(P15, P25),
    get_segment_curve(P25, H), get_segment_curve(H, B)
])
chapter_curves["10-2 二重积分的计算方法"] = build_closed_curve([
    arc["P15C"], get_segment_curve(C, I),
    get_segment_curve(I, P25), get_segment_curve(P25, P15)
])
chapter_curves["10-3 三重积分"] = build_closed_curve([
    get_segment_curve(H, P25), get_segment_curve(P25, P36),
    get_segment_curve(P36, E), get_segment_curve(E, H)
])
chapter_curves["10-4 重积分的应用"] = build_closed_curve([
    get_segment_curve(P25, I), get_segment_curve(I, F),
    get_segment_curve(F, P36), get_segment_curve(P36, P25)
])

# -------- 第十一章 曲线积分与曲面积分（7节） --------
chapter_curves["11-1 对弧长的曲线积分"] = build_closed_curve([
    arc["CP16"], get_segment_curve(P16, P26),
    get_segment_curve(P26, I), get_segment_curve(I, C)
])
chapter_curves["11-2 对坐标的曲线积分"] = build_closed_curve([
    arc["P16P18"], get_segment_curve(P18, P27),
    get_segment_curve(P27, P26), get_segment_curve(P26, P16)
])
chapter_curves["11-3 格林公式及其应用"] = build_closed_curve([
    arc["P18A"], get_segment_curve(A, G),
    get_segment_curve(G, P27), get_segment_curve(P27, P18)
])
chapter_curves["11-4 对面积的曲面积分"] = build_closed_curve([
    get_segment_curve(I, P26), get_segment_curve(P26, P37),
    get_segment_curve(P37, F), get_segment_curve(F, I)
])
chapter_curves["11-5 对坐标的曲面积分"] = build_closed_curve([
    get_segment_curve(P26, P27), get_segment_curve(P27, P39),
    get_segment_curve(P39, P37), get_segment_curve(P37, P26)
])
chapter_curves["11-6/7 高斯公式·斯托克斯公式"] = build_closed_curve([
    get_segment_curve(P27, G), get_segment_curve(G, D),
    get_segment_curve(D, P39), get_segment_curve(P39, P27)
])

# -------- 第十二章 无穷级数（7节） --------
chapter_curves["12-1 常数项级数的概念和性质"] = build_closed_curve([
    get_segment_curve(F, P38), get_segment_curve(P38, P312), get_segment_curve(P312, F)
])
chapter_curves["12-2 常数项级数的审敛法"] = build_closed_curve([
    get_segment_curve(F, P312), get_segment_curve(P312, P36), get_segment_curve(P36, F)
])
chapter_curves["12-3 幂级数"] = build_closed_curve([
    get_segment_curve(D, P31), get_segment_curve(P31, Df), get_segment_curve(Df, F)
])
chapter_curves["12-4 函数展开成幂级数"] = build_closed_curve([
    get_segment_curve(P31, Df), get_segment_curve(Df, P38),
    get_segment_curve(P38, P310), get_segment_curve(P310, P32),
    get_segment_curve(P32, P31)
])
chapter_curves["12-5 函数的幂级数展开式的应用"] = build_closed_curve([
    get_segment_curve(P310, P311), get_segment_curve(P311, P33),
    get_segment_curve(P33, P32), get_segment_curve(P32, P310)
])
chapter_curves["12-6 傅里叶级数"] = build_closed_curve([
    get_segment_curve(P311, P36), get_segment_curve(P36, Ef),
    get_segment_curve(Ef, P34), get_segment_curve(P34, P33),
    get_segment_curve(P33, P311)
])
chapter_curves["12-7 一般周期函数的傅里叶级数"] = build_closed_curve([
    get_segment_curve(Ef, E), get_segment_curve(E, P34), get_segment_curve(P34, Ef)
])

# ===================== 7. 3D绘图：球面 + 曲线 + 仅端点标注 + 图例外置 =====================
fig = plt.figure(figsize=(16, 10), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# 绘制半透明球面
theta = np.linspace(0, 2*np.pi, 70)
phi = np.linspace(0, np.pi, 70)
theta_grid, phi_grid = np.meshgrid(theta, phi)
x_sph = R * np.sin(phi_grid) * np.cos(theta_grid)
y_sph = R * np.sin(phi_grid) * np.sin(theta_grid)
z_sph = R * np.cos(phi_grid)

ax.plot_surface(
    x_sph, y_sph, z_sph,
    color="lightgray", alpha=0.18,
    linewidth=0, antialiased=True
)

# 绘制所有章节曲线
colors = plt.cm.tab20(np.linspace(0, 1, len(chapter_curves)))
for idx, (name, curve) in enumerate(chapter_curves.items()):
    ax.plot(
        curve[:, 0], curve[:, 1], curve[:, 2],
        color=colors[idx], linewidth=1.3, label=name
    )

# 绘制【仅线段/圆弧端点】标注
label_points = collect_all_label_points()
for point_name, coord in label_points.items():
    projected_pt = point_to_sphere(coord)
    ax.text(
        projected_pt[0], projected_pt[1], projected_pt[2],
        point_name,
        color='black', fontsize=7, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        zorder=10
    )

# 隐藏坐标轴、网格
ax.axis("off")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 视角
ax.view_init(elev=30, azim=45)

# 图例外置、多列、不遮挡球面
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=6,
    ncol=2,
    framealpha=0.8
)

plt.tight_layout()
plt.subplots_adjust(right=0.78)

plt.show()