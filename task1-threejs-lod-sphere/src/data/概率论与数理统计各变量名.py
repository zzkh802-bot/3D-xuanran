import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 球面参数设置
R = 1.0  # 球面半径
r1 = 0.8  # 上层圆半径
r2 = 0.3  # 下层小圆半径
h1 = np.sqrt(R ** 2 - r1 ** 2)  # 上层z坐标
h2 = np.sqrt(R ** 2 - r1 ** 2)  # 下层z坐标绝对值（与上层对称）

# 存储已显示的点，避免重复
displayed_points = set()

# 球面投影函数（文档中特别强调的投影方式）
def project_to_sphere(x, y, z):
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (R / norm) * x, (R / norm) * y, (R / norm) * z

# 绘制端点 + 标注变量名
def draw_point_with_label(ax, point, label, color='black', size=6):
    # 浮点精度处理，防止重复判断失误
    point_tuple = tuple(round(p, 6) for p in point)
    if point_tuple not in displayed_points:
        x, y, z = project_to_sphere(*point)
        # 绘制散点
        ax.scatter(x, y, z, color=color, s=size, zorder=10, edgecolors='white', linewidth=0.5)
        # 标注变量名称，偏移一点防止遮挡
        ax.text(x+0.02, y+0.02, z+0.02, label, fontsize=7, color='black', zorder=11)
        displayed_points.add(point_tuple)

# 绘制球面背景
def draw_sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = R * np.cos(u) * np.sin(v)
    y = R * np.sin(u) * np.sin(v)
    z = R * np.cos(v)
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.1, linewidth=0)

# 绘制上层圆弧（z=h1平面）
def draw_upper_arc(ax, t_start, t_end, color='blue', linewidth=2):
    t = np.linspace(t_start, t_end, 100)
    x = r1 * np.cos(t)
    y = r1 * np.sin(t)
    z = np.full_like(t, h1)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color=color, linewidth=linewidth)

# 绘制线段（两点之间）
def draw_line_segment(ax, p1, p2, color='red', linewidth=1.5):
    t = np.linspace(0, 1, 100)
    x = p1[0] * (1 - t) + p2[0] * t
    y = p1[1] * (1 - t) + p2[1] * t
    z = p1[2] * (1 - t) + p2[2] * t
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color=color, linewidth=linewidth)

# 定义所有关键点坐标
O1 = (0, 0, h1)
O2 = (0, 0, -h2)
# 上层圆上的端点A,B,C,D
A = (r1 * np.cos(-3 * np.pi / 4), r1 * np.sin(-3 * np.pi / 4), h1)
B = (r1 * np.cos(-np.pi / 4), r1 * np.sin(-np.pi / 4), h1)
C = (r1 * np.cos(np.pi / 4), r1 * np.sin(np.pi / 4), h1)
D = (r1 * np.cos(3 * np.pi / 4), r1 * np.sin(3 * np.pi / 4), h1)
# 上层圆上的细分点P11-P15
t1 = -np.pi / 12 - np.pi / 2  # -7π/12
t2 = np.pi / 12 - np.pi / 2  # -5π/12
t3 = 0
t4 = np.pi / 2
t5 = np.pi
P11 = (r1 * np.cos(t1), r1 * np.sin(t1), h1)
P12 = (r1 * np.cos(t2), r1 * np.sin(t2), h1)
P13 = (r1 * np.cos(t3), r1 * np.sin(t3), h1)
P14 = (r1 * np.cos(t4), r1 * np.sin(t4), h1)
P15 = (r1 * np.cos(t5), r1 * np.sin(t5), h1)
# 下层大圆上的端点E,F,G,H
E = (-np.sqrt(2) / 2 * r1, -np.sqrt(2) / 2 * r1, -h2)
F = (np.sqrt(2) / 2 * r1, -np.sqrt(2) / 2 * r1, -h2)
G = (np.sqrt(2) / 2 * r1, np.sqrt(2) / 2 * r1, -h2)
H = (-np.sqrt(2) / 2 * r1, np.sqrt(2) / 2 * r1, -h2)
EF=((E[0] + F[0]) / 2, (E[1] + F[1]) / 2, (E[2] + F[2]) / 2)
GH=((G[0] + H[0]) / 2, (G[1] + H[1]) / 2, (G[2] + H[2]) / 2)
FG=((G[0] + F[0]) / 2, (G[1] + F[1]) / 2, (G[2] + F[2]) / 2)
# 中层点I,J,K,L
I = ((A[0] + E[0]) / 2, (A[1] + E[1]) / 2, (A[2] + E[2]) / 2)
J = ((B[0] + F[0]) / 2, (B[1] + F[1]) / 2, (B[2] + F[2]) / 2)
K = ((C[0] + G[0]) / 2, (C[1] + G[1]) / 2, (C[2] + G[2]) / 2)
L = ((D[0] + H[0]) / 2, (D[1] + H[1]) / 2, (D[2] + H[2]) / 2)
# 下层细分点P31-P37
P31 = (-np.sqrt(2) / 6 * r1, -np.sqrt(2) / 2 * r1, -h2)
P32 = (np.sqrt(2) / 6 * r1, -np.sqrt(2) / 2 * r1, -h2)
P33 = (np.sqrt(2) / 2 * r1, -np.sqrt(2) / 6 * r1, -h2)
P34 = (np.sqrt(2) / 2 * r1, np.sqrt(2) / 6 * r1, -h2)
P35 = (np.sqrt(2) / 6 * r1, np.sqrt(2) / 2 * r1, -h2)
P36 = (-np.sqrt(2) / 6 * r1, np.sqrt(2) / 2 * r1, -h2)
P37 = (-np.sqrt(2) / 2 * r1, 0, -h2)
# 中层细分点P21-P28
P21 = (2 * I[0] + J[0]) / 3, (2 * I[1] + J[1]) / 3, (2 * I[2] + J[2]) / 3  # 三等分点靠I近
P22 = (I[0] + 2 * J[0]) / 3, (I[1] + 2 * J[1]) / 3, (I[2] + 2 * J[2]) / 3  # 三等分点靠J近
P23 = (2 * J[0] + K[0]) / 3, (2 * J[1] + K[1]) / 3, (2 * J[2] + K[2]) / 3  # 三等分点靠J近
P25 = (J[0] + 2 * K[0]) / 3, (J[1] + 2 * K[1]) / 3, (J[2] + 2 * K[2]) / 3  # 三等分点靠K近
P24 = ((J[0] + K[0]) / 2, (J[1] + K[1]) / 2, (J[2] + K[2]) / 2)  # JK中点
P26 = (2 * K[0] + L[0]) / 3, (2 * K[1] + L[1]) / 3, (2 * K[2] + L[2]) / 3  # 三等分点靠K近
P28 = (K[0] + 2 * L[0]) / 3, (K[1] + 2 * L[1]) / 3, (K[2] + 2 * L[2]) / 3  # 三等分点靠L近
P27 = ((K[0] + L[0]) / 2, (K[1] + L[1]) / 2, (K[2] + L[2]) / 2)  # KL中点
# 下层小圆上的点P41-P46
P41 = ((EF[0] + P37[0]) / 2, (EF[1] + P37[1]) / 2, (EF[2] + P37[2]) / 2)
P42 = ((FG[0] + GH[0]) / 2, (FG[1] + GH[1]) / 2, (FG[2] + GH[2]) / 2)
P43 = (-r2 * np.sqrt(2) / 2, -r2 * np.sqrt(2) / 2, -h2)
P44 = (r2 * np.sqrt(2) / 2, r2 * np.sqrt(2) / 2, -h2)
P45 = (r2 * np.sqrt(2) / 2, -r2 * np.sqrt(2) / 2, -h2)
P46 = (-r2 * np.sqrt(2) / 2, r2 * np.sqrt(2) / 2, -h2)

# 绘制所有章节边界
def draw_all_sections(ax):
    # 第一章：概率论的基本概念
    # 第一节：随机事件
    draw_line_segment(ax, I, P21, color='#FF6B6B')
    draw_line_segment(ax, P21, P31, color='#FF6B6B')
    draw_line_segment(ax, P31, E, color='#FF6B6B')
    draw_line_segment(ax, E, I, color='#FF6B6B')
    # 第二节：频率与概率
    draw_line_segment(ax, P21, P22, color='#4ECDC4')
    draw_line_segment(ax, P22, P32, color='#4ECDC4')
    draw_line_segment(ax, P32, P31, color='#4ECDC4')
    draw_line_segment(ax, P31, P21, color='#4ECDC4')
    # 第三节：古典概型与几何概型
    draw_line_segment(ax, P22, J, color='#45B7D1')
    draw_line_segment(ax, J, F, color='#45B7D1')
    draw_line_segment(ax, F, P32, color='#45B7D1')
    draw_line_segment(ax, P32, P22, color='#45B7D1')
    # 第四节：条件概率
    draw_upper_arc(ax, t2, -np.pi / 4, color='#96CEB4')
    draw_line_segment(ax, B, J, color='#96CEB4')
    draw_line_segment(ax, J, P22, color='#96CEB4')
    draw_line_segment(ax, P22, P12, color='#96CEB4')
    # 第五节：全概率公式与贝叶斯公式
    draw_upper_arc(ax, t1, t2, color='#FFEAA7')
    draw_line_segment(ax, P12, P22, color='#FFEAA7')
    draw_line_segment(ax, P22, P21, color='#FFEAA7')
    draw_line_segment(ax, P21, P11, color='#FFEAA7')
    # 第六节：两两独立、相互独立
    draw_upper_arc(ax, -3 * np.pi / 4, t1, color='#DDA0DD')
    draw_line_segment(ax, P11, P21, color='#DDA0DD')
    draw_line_segment(ax, P21, I, color='#DDA0DD')
    draw_line_segment(ax, I, A, color='#DDA0DD')
    # 第二章：随机变量及其分布
    # 第一节：随机变量
    draw_line_segment(ax, J, P23, color='#FF8C00')
    draw_line_segment(ax, P23, P33, color='#FF8C00')
    draw_line_segment(ax, P33, F, color='#FF8C00')
    draw_line_segment(ax, F, J, color='#FF8C00')
    # 第二节：离散型随机变量及其分布律
    draw_line_segment(ax, P23, P25, color='#20B2AA')
    draw_line_segment(ax, P25, P34, color='#20B2AA')
    draw_line_segment(ax, P34, P33, color='#20B2AA')
    draw_line_segment(ax, P33, P23, color='#20B2AA')
    # 第三节：随机变量的分布函数
    draw_line_segment(ax, P25, K, color='#87CEEB')
    draw_line_segment(ax, K, G, color='#87CEEB')
    draw_line_segment(ax, G, P34, color='#87CEEB')
    draw_line_segment(ax, P34, P25, color='#87CEEB')
    # 第四节：随机变量的函数的分布
    draw_upper_arc(ax, 0, np.pi / 4, color='#F0E68C')
    draw_line_segment(ax, C, K, color='#F0E68C')
    draw_line_segment(ax, K, P24, color='#F0E68C')
    draw_line_segment(ax, P24, P13, color='#F0E68C')
    # 第五节：连续型随机变量及其概率密度
    draw_upper_arc(ax, -np.pi / 4, 0, color='#FA8072')
    draw_line_segment(ax, P13, P24, color='#FA8072')
    draw_line_segment(ax, P24, J, color='#FA8072')
    draw_line_segment(ax, J, B, color='#FA8072')
    # 第三章：多维随机变量及其分布
    # 第一节：二维随机变量
    draw_line_segment(ax, K, P26, color='#9370DB')
    draw_line_segment(ax, P26, P35, color='#9370DB')
    draw_line_segment(ax, P35, G, color='#9370DB')
    draw_line_segment(ax, G, K, color='#9370DB')
    # 第二节：边缘分布
    draw_line_segment(ax, P26, P28, color='#3CB371')
    draw_line_segment(ax, P28, P36, color='#3CB371')
    draw_line_segment(ax, P36, P35, color='#3CB371')
    draw_line_segment(ax, P35, P26, color='#3CB371')
    # 第三节：条件分布
    draw_line_segment(ax, P28, L, color='#FF69B4')
    draw_line_segment(ax, L, H, color='#FF69B4')
    draw_line_segment(ax, H, P36, color='#FF69B4')
    draw_line_segment(ax, P36, P28, color='#FF69B4')
    # 第四节：两个常用的二维分布
    draw_upper_arc(ax, np.pi / 4, np.pi / 2, color='#CD853F')
    draw_line_segment(ax, P14, P27, color='#CD853F')
    draw_line_segment(ax, P27, K, color='#CD853F')
    draw_line_segment(ax, K, C, color='#CD853F')
    # 第五节：两个随机变量的函数的分布
    draw_upper_arc(ax, np.pi / 2, 3 * np.pi / 4, color='#6495ED')
    draw_line_segment(ax, D, L, color='#6495ED')
    draw_line_segment(ax, L, P27, color='#6495ED')
    draw_line_segment(ax, P27, P14, color='#6495ED')
    # 第四章：随机变量的数字特征
    # 第一节：数学期望
    draw_upper_arc(ax, -3 * np.pi / 4, -np.pi / 4, color='#DC143C')
    draw_line_segment(ax, B, O1, color='#DC143C')
    draw_line_segment(ax, O1, A, color='#DC143C')
    # 第二节：方差
    draw_upper_arc(ax, -np.pi / 4, np.pi / 4, color='#00CED1')
    draw_line_segment(ax, C, O1, color='#00CED1')
    draw_line_segment(ax, O1, B, color='#00CED1')
    # 第三节：协方差与相关系数
    draw_upper_arc(ax, np.pi / 4, 3 * np.pi / 4, color='#FFD700')
    draw_line_segment(ax, D, O1, color='#FFD700')
    draw_line_segment(ax, O1, C, color='#FFD700')
    # 第四节：矩、协方差矩阵
    draw_upper_arc(ax, 3 * np.pi / 4, 5 * np.pi / 4, color='#8A2BE2')
    draw_line_segment(ax, A, O1, color='#8A2BE2')
    draw_line_segment(ax, O1, D, color='#8A2BE2')
    # 第五章：大数定律和中心极限定理
    # 第一节：大数定律
    draw_upper_arc(ax, 3 * np.pi / 4, np.pi, color='#00FA9A')
    draw_line_segment(ax, P15, P37, color='#00FA9A')
    draw_line_segment(ax, P37, H, color='#00FA9A')
    draw_line_segment(ax, H, D, color='#00FA9A')
    # 第二节：中心极限定理
    draw_upper_arc(ax, np.pi, 5 * np.pi / 4, color='#FF4500')
    draw_line_segment(ax, A, E, color='#FF4500')
    draw_line_segment(ax, E, P37, color='#FF4500')
    draw_line_segment(ax, P37, P15, color='#FF4500')
    # 第六章：统计量与抽样分布
    # 第一节：随机样本
    draw_line_segment(ax, E, P37, color='#7B68EE')
    draw_line_segment(ax, P37, EF, color='#7B68EE')
    draw_line_segment(ax, EF, E, color='#7B68EE')
    # 第二节：抽样分布
    draw_line_segment(ax, P37, P41, color='#00FF7F')
    draw_line_segment(ax, P41, P43, color='#00FF7F')
    # P46P43弧的逆
    t = np.linspace(5 * np.pi / 4, 3 * np.pi / 4, 100)
    x = r2 * np.cos(t)
    y = r2 * np.sin(t)
    z = np.full_like(t, -h2)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color='#00FF7F', linewidth=1.5)
    draw_line_segment(ax, P46, H, color='#00FF7F')
    draw_line_segment(ax, H, P37, color='#00FF7F')
    # 第三节：正态总体的抽样分布
    draw_line_segment(ax, P41, EF, color='#FF1493')
    draw_line_segment(ax, EF, F, color='#FF1493')
    draw_line_segment(ax, F, P45, color='#FF1493')
    # P43P45弧的逆
    t = np.linspace(-np.pi / 4, -3 * np.pi / 4, 100)
    x = r2 * np.cos(t)
    y = r2 * np.sin(t)
    z = np.full_like(t, -h2)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color='#FF1493', linewidth=1.5)
    draw_line_segment(ax, P43, P41, color='#FF1493')
    # 第七章：参数估计
    # 第一节点估计
    draw_line_segment(ax, G, FG, color='#00BFFF')
    draw_line_segment(ax, FG, GH, color='#00BFFF')
    draw_line_segment(ax, GH, G, color='#00BFFF')
    # 第二节估计量的评选标准
    draw_line_segment(ax, FG, P42, color='#FF6347')
    draw_line_segment(ax, P42, P44, color='#FF6347')
    # P45P44弧的逆
    t = np.linspace(np.pi / 4, -np.pi / 4, 100)
    x = r2 * np.cos(t)
    y = r2 * np.sin(t)
    z = np.full_like(t, -h2)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color='#FF6347', linewidth=1.5)
    draw_line_segment(ax, P45, F, color='#FF6347')
    draw_line_segment(ax, F, FG, color='#FF6347')
    # 第三节区间估计
    draw_line_segment(ax, P42, GH, color='#7FFF00')
    draw_line_segment(ax, GH, H, color='#7FFF00')
    # P44P46弧的逆
    t = np.linspace(3 * np.pi / 4, np.pi / 4, 100)
    x = r2 * np.cos(t)
    y = r2 * np.sin(t)
    z = np.full_like(t, -h2)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color='#7FFF00', linewidth=1.5)
    draw_line_segment(ax, P45, F, color='#7FFF00')
    # 第八章：假设检验
    # 第一节假设检验的基本思想
    # P46P43弧
    t = np.linspace(3 * np.pi / 4, 5 * np.pi / 4, 100)
    x = r2 * np.cos(t)
    y = r2 * np.sin(t)
    z = np.full_like(t, -h2)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color='#8B008B', linewidth=1.5)
    # 第二节正态总体均值和方差的假设检验
    # P43P46弧
    t = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 100)
    x = r2 * np.cos(t)
    y = r2 * np.sin(t)
    z = np.full_like(t, -h2)
    x_sph, y_sph, z_sph = project_to_sphere(x, y, z)
    ax.plot(x_sph, y_sph, z_sph, color='#008080', linewidth=1.5)
    draw_line_segment(ax, P46, O2, color='#008080')
    draw_line_segment(ax, O2, P43, color='#008080')

# 创建图形
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制球面和所有边界
draw_sphere(ax)
draw_all_sections(ax)

# 所有端点 + 对应变量名（一一绑定，自动去重）
point_label_list = [
    (O1, "O1"), (O2, "O2"),
    (A, "A"), (B, "B"), (C, "C"), (D, "D"),
    (P11, "P11"), (P12, "P12"), (P13, "P13"), (P14, "P14"), (P15, "P15"),
    (E, "E"), (F, "F"), (G, "G"), (H, "H"),
    (EF, "EF"), (FG, "FG"), (GH, "GH"),
    (I, "I"), (J, "J"), (K, "K"), (L, "L"),
    (P31, "P31"), (P32, "P32"), (P33, "P33"), (P34, "P34"),
    (P35, "P35"), (P36, "P36"), (P37, "P37"),
    (P21, "P21"), (P22, "P22"), (P23, "P23"), (P24, "P24"),
    (P25, "P25"), (P26, "P26"), (P27, "P27"), (P28, "P28"),
    (P41, "P41"), (P42, "P42"), (P43, "P43"), (P44, "P44"),
    (P45, "P45"), (P46, "P46")
]

# 批量绘制端点并标注变量名
for pt, lab in point_label_list:
    draw_point_with_label(ax, pt, lab)

# 隐藏坐标轴和网格
ax.set_axis_off()
ax.grid(False)
# 设置视角
ax.view_init(elev=30, azim=45)
# 调整布局
plt.tight_layout()
# 显示图形
plt.show()