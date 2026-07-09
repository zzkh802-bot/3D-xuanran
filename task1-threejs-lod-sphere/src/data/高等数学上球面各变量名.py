import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ===================== 全局参数设置 =====================
R = 1.0  # 球面半径（严格按照文档归一化公式）
h1 = 0.6  # 上平面z坐标
h2 = 0.48  # 下平面z坐标
r1 = np.sqrt(R ** 2 - h1 ** 2)  # 最外层圆半径（上平面）
r2 = r1 * 0.6  # 中间圆半径（上平面）
r3 = r1 * 0.3  # 最内层圆半径（上平面）
num_points = 200  # 曲线采样点数
alpha_sphere = 0.2  # 球面透明度
line_width = 2.5  # 边界曲线宽度
# ===================== 核心投影函数（文档特别强调） =====================
def project_to_sphere(x, y, z, R):
    """
    严格按照文档公式将空间点投影到半径为R的球面上
    公式: x' = R/√(x²+y²+z²) * x
         y' = R/√(x²+y²+z²) * y
         z' = R/√(x²+y²+z²) * z
    """
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x_sphere = R * x / norm
    y_sphere = R * y / norm
    z_sphere = R * z / norm
    return x_sphere, y_sphere, z_sphere
# ===================== 曲线生成函数（严格遵循文档参数化） =====================
def generate_arc(r, t_start, t_end, z_plane, R, num=num_points):
    """生成平面圆弧并投影到球面"""
    t = np.linspace(t_start, t_end, num)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = np.full_like(x, z_plane)
    return project_to_sphere(x, y, z, R)
def generate_line(p1, p2, R, num=num_points):
    """
    严格按照文档线段参数化公式生成线段并投影到球面
    公式: P(t) = P1*(1-t) + P2*t, 0 ≤ t ≤ 1
    p1, p2: 三维坐标元组 (x,y,z)
    """
    t = np.linspace(0, 1, num)
    x = p1[0] * (1 - t) + p2[0] * t
    y = p1[1] * (1 - t) + p2[1] * t
    z = p1[2] * (1 - t) + p2[2] * t
    return project_to_sphere(x, y, z, R)
# ===================== 定义所有关键点（三维坐标） =====================
# 上平面 (z=h1) 关键点
points_upper = {
    'A': (r1 * np.cos(-5 * np.pi / 6), r1 * np.sin(-5 * np.pi / 6), h1),
    'B': (r1 * np.cos(-np.pi / 2), r1 * np.sin(-np.pi / 2), h1),
    'C': (r1 * np.cos(-np.pi / 6), r1 * np.sin(-np.pi / 6), h1),
    'D': (r1 * np.cos(np.pi / 6), r1 * np.sin(np.pi / 6), h1),
    'E': (r1 * np.cos(np.pi / 2), r1 * np.sin(np.pi / 2), h1),
    'F': (r1 * np.cos(5 * np.pi / 6), r1 * np.sin(5 * np.pi / 6), h1),
    'P11': (r1 * np.cos(0), r1 * np.sin(0), h1),
    'P12': (r1 * np.cos(np.pi), r1 * np.sin(np.pi), h1),
    'P13': (r1 * np.cos(-2 * np.pi / 3), r1 * np.sin(-2 * np.pi / 3), h1),
    'P14': (r1 * np.cos(-np.pi / 2 + np.pi / 9), r1 * np.sin(-np.pi / 2 + np.pi / 9), h1),
    'P15': (r1 * np.cos(-np.pi / 2 + 2 * np.pi / 9), r1 * np.sin(-np.pi / 2 + 2 * np.pi / 9), h1),
    'P16': (r1 * np.cos(0), r1 * np.sin(0), h1),
    'P17': (r1 * np.cos(np.pi / 3), r1 * np.sin(np.pi / 3), h1),
    'P18': (r1 * np.cos(2 * np.pi / 3), r1 * np.sin(2 * np.pi / 3), h1),
    'P19': (r1 * np.cos(5 * np.pi / 6 + np.pi / 9), r1 * np.sin(5 * np.pi / 6 + np.pi / 9), h1),
    'P110': (r1 * np.cos(5 * np.pi / 6 + 2 * np.pi / 9), r1 * np.sin(5 * np.pi / 6 + 2 * np.pi / 9), h1),
    'P21': (r2 * np.cos(0), r2 * np.sin(0), h1),
    'P22': (r2 * np.cos(np.pi / 3), r2 * np.sin(np.pi / 3), h1),
    'P23': (r2 * np.cos(2 * np.pi / 3), r2 * np.sin(2 * np.pi / 3), h1),
    'P24': (r2 * np.cos(np.pi), r2 * np.sin(np.pi), h1),
    'P25': (r2 * np.cos(4 * np.pi / 3), r2 * np.sin(4 * np.pi / 3), h1),
    'P26': (r2 * np.cos(-np.pi / 2), r2 * np.sin(-np.pi / 2), h1),
    'P27': (r2 * np.cos(-np.pi / 3), r2 * np.sin(-np.pi / 3), h1),
    'P31': (r3 * np.cos(0), r3 * np.sin(0), h1),
    'P32': (r3 * np.cos(np.pi / 3), r3 * np.sin(np.pi / 3), h1),
    'P33': (r3 * np.cos(2 * np.pi / 3), r3 * np.sin(2 * np.pi / 3), h1),
    'P34': (r3 * np.cos(np.pi), r3 * np.sin(np.pi), h1),
    'P35': (r3 * np.cos(4 * np.pi / 3), r3 * np.sin(4 * np.pi / 3), h1),
    'P36': (r3 * np.cos(-np.pi / 3), r3 * np.sin(-np.pi / 3), h1),
}
# 下平面 (z=-h2) 关键点（严格按照文档坐标公式）
r_lower = np.sqrt(R ** 2 - h2 ** 2)
points_lower = {
    'G': (-np.sqrt(3) / 2 * r_lower, -1 / 2 * r_lower, -h2),
    'H': (0, -1 / 2 *r_lower, -h2),
    'I': (np.sqrt(3) / 2 * r_lower, -1 / 2 * r_lower, -h2),
    'J': (np.sqrt(3) / 4 * r_lower, 1 / 4 * r_lower, -h2),
    'K': (0, r_lower, -h2),
    'L': (-np.sqrt(3) / 4 * r_lower, 1 / 4 * r_lower, -h2),
}
# 计算P4系列点（上下对应点的中点）
points_mid4 = {
    'P41': (3/ 7 *(points_upper['A'][0] + 4/ 7 *points_lower['G'][0]) ,
            3/ 7 *(points_upper['A'][1] + 4/ 7 *points_lower['G'][1]) ,
            3/ 7 *(points_upper['A'][2] + 4/ 7 *points_lower['G'][2]) ),
    'P42': (3/ 7*(points_upper['B'][0] +  4/ 7 *points_lower['H'][0]) ,
            3/ 7 *(points_upper['B'][1] +  4/ 7 *points_lower['H'][1]) ,
            3/ 7 *(points_upper['B'][2] +  4/ 7 *points_lower['H'][2]) ),
    'P43': (3/ 7 *(points_upper['C'][0] +  4/ 7 *points_lower['I'][0]) ,
            3/ 7 *(points_upper['C'][1] +  4/ 7 *points_lower['I'][1]) ,
            3/ 7 *(points_upper['C'][2] +  4/ 7 *points_lower['I'][2]) ),
    'P44': (3/ 7 *(points_upper['D'][0] +  4/ 7 *points_lower['J'][0]) ,
            3/ 7 *(points_upper['D'][1] +  4/ 7 *points_lower['J'][1]) ,
            3/ 7 *(points_upper['D'][2] +  4/ 7 *points_lower['J'][2]) ),
    'P45': (3/ 7 *(points_upper['E'][0] +  4/ 7 *points_lower['K'][0]) ,
            3/ 7 *(points_upper['E'][1] +  4/ 7 *points_lower['K'][1]) ,
            3/ 7 *(points_upper['E'][2] +  4/ 7 *points_lower['K'][2]) ),
    'P46': (3/ 7 *(points_upper['F'][0] +  4/ 7 *points_lower['L'][0]) ,
            3/ 7 *(points_upper['F'][1] +  4/ 7 *points_lower['L'][1]) ,
            3/ 7 *(points_upper['F'][2] +  4/ 7 *points_lower['L'][2]) ),
}
# 计算P5系列点（严格按照文档三等分/四等分定义）
points_mid5 = {
    'P51': (2 / 3 * points_mid4['P41'][0] + 1 / 3 * points_mid4['P42'][0],
            2 / 3 * points_mid4['P41'][1] + 1 / 3 * points_mid4['P42'][1],
            2 / 3 * points_mid4['P41'][2] + 1 / 3 * points_mid4['P42'][2]),
    'P53': (1 / 3 * points_mid4['P41'][0] + 2 / 3 * points_mid4['P42'][0],
            1 / 3 * points_mid4['P41'][1] + 2 / 3 * points_mid4['P42'][1],
            1 / 3 * points_mid4['P41'][2] + 2 / 3 * points_mid4['P42'][2]),
    'P52': (1 / 2 * points_mid4['P41'][0] + 1 / 2 * points_mid4['P42'][0],
            1 / 2 * points_mid4['P41'][1] + 1 / 2 * points_mid4['P42'][1],
            1 / 2 * points_mid4['P41'][2] + 1 / 2 * points_mid4['P42'][2]),  # 文档隐含的中点
    'P54': (3 / 4 * points_mid4['P42'][0] + 1 / 4 * points_mid4['P43'][0],
            3 / 4 * points_mid4['P42'][1] + 1 / 4 * points_mid4['P43'][1],
            3 / 4 * points_mid4['P42'][2] + 1 / 4 * points_mid4['P43'][2]),
    'P55': (2 / 3 * points_mid4['P42'][0] + 1 / 3 * points_mid4['P43'][0],
            2 / 3 * points_mid4['P42'][1] + 1 / 3 * points_mid4['P43'][1],
            2 / 3 * points_mid4['P42'][2] + 1 / 3 * points_mid4['P43'][2]),
    'P56': (1 / 2 * points_mid4['P42'][0] + 1 / 2 * points_mid4['P43'][0],
            1 / 2 * points_mid4['P42'][1] + 1 / 2 * points_mid4['P43'][1],
            1 / 2 * points_mid4['P42'][2] + 1 / 2 * points_mid4['P43'][2]),
    'P57': (1 / 3 * points_mid4['P42'][0] + 2 / 3 * points_mid4['P43'][0],
            1 / 3 * points_mid4['P42'][1] + 2 / 3 * points_mid4['P43'][1],
            1 / 3 * points_mid4['P42'][2] + 2 / 3 * points_mid4['P43'][2]),
    'P58': (1 / 4 * points_mid4['P42'][0] + 3 / 4 * points_mid4['P43'][0],
            1 / 4 * points_mid4['P42'][1] + 3 / 4 * points_mid4['P43'][1],
            1 / 4 * points_mid4['P42'][2] + 3 / 4 * points_mid4['P43'][2]),
    'P59': (1 / 2 * points_mid4['P43'][0] + 1 / 2 * points_mid4['P44'][0],
            1 / 2 * points_mid4['P43'][1] + 1 / 2 * points_mid4['P44'][1],
            1 / 2 * points_mid4['P43'][2] + 1 / 2 * points_mid4['P44'][2]),
    'P510': (1 / 2 * points_mid4['P44'][0] + 1 / 2 * points_mid4['P45'][0],
             1 / 2 * points_mid4['P44'][1] + 1 / 2 * points_mid4['P45'][1],
             1 / 2 * points_mid4['P44'][2] + 1 / 2 * points_mid4['P45'][2]),
    'P511': (2 / 3 * points_mid4['P45'][0] + 1 / 3 * points_mid4['P46'][0],
             2 / 3 * points_mid4['P45'][1] + 1 / 3 * points_mid4['P46'][1],
             2 / 3 * points_mid4['P45'][2] + 1 / 3 * points_mid4['P46'][2]),
    'P512': (1 / 2 * points_mid4['P45'][0] + 1 / 2 * points_mid4['P46'][0],
             1 / 2 * points_mid4['P45'][1] + 1 / 2 * points_mid4['P46'][1],
             1 / 2 * points_mid4['P45'][2] + 1 / 2 * points_mid4['P46'][2]),
    'P513': (1 / 3 * points_mid4['P45'][0] + 2 / 3 * points_mid4['P46'][0],
             1 / 3 * points_mid4['P45'][1] + 2 / 3 * points_mid4['P46'][1],
             1 / 3 * points_mid4['P45'][2] + 2 / 3 * points_mid4['P46'][2]),
    'P514': (3 / 4 * points_mid4['P46'][0] + 1 / 4 * points_mid4['P41'][0],
             3 / 4 * points_mid4['P46'][1] + 1 / 4 * points_mid4['P41'][1],
             3 / 4 * points_mid4['P46'][2] + 1 / 4 * points_mid4['P41'][2]),
    'P515': (2 / 3 * points_mid4['P46'][0] + 1 / 3 * points_mid4['P41'][0],
             2 / 3 * points_mid4['P46'][1] + 1 / 3 * points_mid4['P41'][1],
             2 / 3 * points_mid4['P46'][2] + 1 / 3 * points_mid4['P41'][2]),
    'P516': (1 / 2 * points_mid4['P46'][0] + 1 / 2 * points_mid4['P41'][0],
             1 / 2 * points_mid4['P46'][1] + 1 / 2 * points_mid4['P41'][1],
             1 / 2 * points_mid4['P46'][2] + 1 / 2 * points_mid4['P41'][2]),
    'P517': (1 / 3 * points_mid4['P46'][0] + 2 / 3 * points_mid4['P41'][0],
             1 / 3 * points_mid4['P46'][1] + 2 / 3 * points_mid4['P41'][1],
             1 / 3 * points_mid4['P46'][2] + 2 / 3 * points_mid4['P41'][2]),
    'P518': (1 / 4 * points_mid4['P46'][0] + 3 / 4 * points_mid4['P41'][0],
             1 / 4 * points_mid4['P46'][1] + 3 / 4 * points_mid4['P41'][1],
             1 / 4 * points_mid4['P46'][2] + 3 / 4 * points_mid4['P41'][2]),
}
# 计算P6系列点（严格按照文档三等分/四等分定义）
points_mid6 = {
    'P61': (2 / 3 * points_lower['G'][0] + 1 / 3 * points_lower['H'][0],
            2 / 3 * points_lower['G'][1] + 1 / 3 * points_lower['H'][1],
            2 / 3 * points_lower['G'][2] + 1 / 3 * points_lower['H'][2]),
    'P62': (1 / 3 * points_lower['G'][0] + 2 / 3 * points_lower['H'][0],
            1 / 3 * points_lower['G'][1] + 2 / 3 * points_lower['H'][1],
            1 / 3 * points_lower['G'][2] + 2 / 3 * points_lower['H'][2]),
    'P63': (3 / 4 * points_lower['H'][0] + 1 / 4 * points_lower['I'][0],
            3 / 4 * points_lower['H'][1] + 1 / 4 * points_lower['I'][1],
            3 / 4 * points_lower['H'][2] + 1 / 4 * points_lower['I'][2]),
    'P64': (1 / 2 * points_lower['H'][0] + 1 / 2 * points_lower['I'][0],
            1 / 2 * points_lower['H'][1] + 1 / 2 * points_lower['I'][1],
            1 / 2 * points_lower['H'][2] + 1 / 2 * points_lower['I'][2]),
    'P65': (1 / 4 * points_lower['H'][0] + 3 / 4 * points_lower['I'][0],
            1 / 4 * points_lower['H'][1] + 3 / 4 * points_lower['I'][1],
            1 / 4 * points_lower['H'][2] + 3 / 4 * points_lower['I'][2]),
    'P66': (1 / 2 * points_lower['I'][0] + 1 / 2 * points_lower['J'][0],
            1 / 2 * points_lower['I'][1] + 1 / 2 * points_lower['J'][1],
            1 / 2 * points_lower['I'][2] + 1 / 2 * points_lower['J'][2]),
    'P67': (1 / 2 * points_lower['J'][0] + 1 / 2 * points_lower['K'][0],
            1 / 2 * points_lower['J'][1] + 1 / 2 * points_lower['K'][1],
            1 / 2 * points_lower['J'][2] + 1 / 2 * points_lower['K'][2]),
    'P68': (2 / 3 * points_lower['K'][0] + 1 / 3 * points_lower['L'][0],
            2 / 3 * points_lower['K'][1] + 1 / 3 * points_lower['L'][1],
            2 / 3 * points_lower['K'][2] + 1 / 3 * points_lower['L'][2]),
    'P69': (1 / 3 * points_lower['K'][0] + 2 / 3 * points_lower['L'][0],
            1 / 3 * points_lower['K'][1] + 2 / 3 * points_lower['L'][1],
            1 / 3 * points_lower['K'][2] + 2 / 3 * points_lower['L'][2]),
    'P610': (3 / 4 * points_lower['L'][0] + 1 / 4 * points_lower['G'][0],
             3 / 4 * points_lower['L'][1] + 1 / 4 * points_lower['G'][1],
             3 / 4 * points_lower['L'][2] + 1 / 4 * points_lower['G'][2]),
    'P611': (1 / 2 * points_lower['L'][0] + 1 / 2 * points_lower['G'][0],
             1 / 2 * points_lower['L'][1] + 1 / 2 * points_lower['G'][1],
             1 / 2 * points_lower['L'][2] + 1 / 2 * points_lower['G'][2]),
    'P612': (1 / 4 * points_lower['L'][0] + 3 / 4 * points_lower['G'][0],
             1 / 4 * points_lower['L'][1] + 3 / 4 * points_lower['G'][1],
             1 / 4 * points_lower['L'][2] + 3 / 4 * points_lower['G'][2]),
}
# 合并所有点到一个字典方便查找
all_points = {**points_upper, **points_lower, **points_mid4, **points_mid5, **points_mid6}
# ===================== 定义各章节边界曲线（严格按照文档顺序） =====================
sections = {
    # 第一章 函数与极限
    "1.1 映射与函数": [
        ('arc', r3, 0, np.pi / 3, h1),
        ('arc', r3, np.pi / 3, 2 * np.pi / 3, h1),
        ('arc', r3, 2 * np.pi / 3, np.pi, h1),
        ('arc', r3, np.pi, 4 * np.pi / 3, h1),
        ('arc', r3, 4 * np.pi / 3, 5 * np.pi / 3, h1),
        ('arc', r3, 5 * np.pi / 3, 2 * np.pi, h1),
    ],
    "1.2 数列的极限": [
        ('arc', r1, -np.pi / 2, 0, h1),
        ('line', 'P11', 'P21'),
        ('arc_rev', r2, -np.pi / 2, 0, h1),
        ('line', 'P26', 'B'),
    ],
    "1.3 函数的极限": [
        ('arc', r1, 0, np.pi, h1),
        ('line', 'P12', 'P24'),
        ('arc_rev', r2, 0, np.pi, h1),
        ('line', 'P21', 'P11'),
    ],
    "1.4 无穷小与无穷大": [
        ('arc', r2, 0, np.pi / 3, h1),
        ('line', 'P22', 'P32'),
        ('arc_rev', r3, 0, np.pi / 3, h1),
        ('line', 'P31', 'P21'),
    ],
    "1.5 极限运算法则": [
        ('arc', r1, -np.pi, -np.pi / 2, h1),
        ('line', 'B', 'P26'),
        ('arc_rev', r2, np.pi, 3 * np.pi / 2, h1),
        ('line', 'P24', 'P12'),
    ],
    "1.6 极限存在准则 两个重要极限": [
        ('arc', r2, np.pi / 3, 2 * np.pi / 3, h1),
        ('line', 'P23', 'P33'),
        ('arc_rev', r3, np.pi / 3, 2 * np.pi / 3, h1),
        ('line', 'P32', 'P22'),
    ],
    "1.7 无穷小的比较": [
        ('arc', r2, -np.pi / 3, 0, h1),
        ('line', 'P21', 'P31'),
        ('arc_rev', r3, -np.pi / 3, 0, h1),
        ('line', 'P36', 'P27'),
    ],
    "1.8 函数的连续性与间断点": [
        ('arc', r2, 2 * np.pi / 3, np.pi, h1),
        ('line', 'P24', 'P34'),
        ('arc_rev', r3, 2 * np.pi / 3, np.pi, h1),
        ('line', 'P33', 'P23'),
    ],
    "1.9 连续函数的运算与初等函数的连续性": [
        ('arc', r2, np.pi, 4 * np.pi / 3, h1),
        ('line', 'P25', 'P35'),
        ('arc_rev', r3, np.pi, 4 * np.pi / 3, h1),
        ('line', 'P34', 'P24'),
    ],
    "1.10 闭区间上连续函数的性质": [
        ('arc', r2, 4 * np.pi / 3, 5 * np.pi / 3, h1),
        ('line', 'P27', 'P36'),
        ('arc_rev', r3, 4 * np.pi / 3, 5 * np.pi / 3, h1),
        ('line', 'P35', 'P25'),
    ],
    # 第二章 导数与微分
    "2.1 导数概念": [
        ('arc', r1, -5 * np.pi / 6, -2 * np.pi / 3, h1),
        ('line', 'P13', 'P52'),
        ('line', 'P52', 'P41'),
        ('line', 'P41', 'A'),
    ],
    "2.2 函数的求导法则": [
        ('arc', r1, -2 * np.pi / 3, -np.pi / 2, h1),
        ('line', 'B', 'P42'),
        ('line', 'P42', 'P52'),
        ('line', 'P52', 'P13'),
    ],
    "2.3 高阶导数": [
        ('line', 'P41', 'P51'),
        ('line', 'P51', 'P61'),
        ('line', 'P61', 'G'),
        ('line', 'G', 'P41'),
    ],
    "2.4 隐函数及参数方程导数": [
        ('line', 'P51', 'P61'),
        ('line', 'P61', 'P62'),
        ('line', 'P62', 'P53'),
        ('line', 'P53', 'P51'),
    ],
    "2.5 函数的微分": [
        ('line', 'P53', 'P62'),
        ('line', 'P62', 'H'),
        ('line', 'H', 'P42'),
        ('line', 'P42', 'P53'),
    ],
    # 第三章 微分中值定理与导数的应用
    "3.1 微分中值定理": [
        ('arc', r1, -np.pi / 2, -np.pi / 2 + np.pi / 9, h1),
        ('line', 'P14', 'P55'),
        ('line', 'P55', 'P42'),
        ('line', 'P42', 'B'),
    ],
    "3.2 洛必达法则": [
        ('arc', r1, -np.pi / 2 + np.pi / 9, -np.pi / 2 + 2 * np.pi / 9, h1),
        ('line', 'P15', 'P57'),
        ('line', 'P57', 'P55'),
        ('line', 'P55', 'P14'),
    ],
    "3.3 泰勒公式": [
        ('arc', r1, -np.pi / 2 + 2 * np.pi / 9, -np.pi / 6, h1),
        ('line', 'C', 'P43'),
        ('line', 'P43', 'P57'),
        ('line', 'P57', 'P15'),
    ],
    "3.4 单调性与凹凸性": [
        ('line', 'P42', 'P54'),
        ('line', 'P54', 'P63'),
        ('line', 'P63', 'H'),
        ('line', 'H', 'P42'),
    ],
    "3.5 极值与最值": [
        ('line', 'P54', 'P56'),
        ('line', 'P56', 'P64'),
        ('line', 'P64', 'P63'),
        ('line', 'P63', 'P54'),
    ],
    "3.6 函数图形的描绘": [
        ('line', 'P56', 'P58'),
        ('line', 'P58', 'P65'),
        ('line', 'P65', 'P64'),
        ('line', 'P64', 'P56'),
    ],
    "3.7 曲率": [
        ('line', 'P58', 'P43'),
        ('line', 'P43', 'I'),
        ('line', 'I', 'P65'),
        ('line', 'P65', 'P58'),
    ],
    # 第四章 不定积分
    "4.1 不定积分概念与性质": [
        ('arc', r1, -np.pi / 6, 0, h1),
        ('line', 'P16', 'P59'),
        ('line', 'P59', 'P43'),
        ('line', 'P43', 'C'),
    ],
    "4.2 换元积分法": [
        ('arc', r1, 0, np.pi / 6, h1),
        ('line', 'D', 'P44'),
        ('line', 'P44', 'P59'),
        ('line', 'P59', 'P16'),
    ],
    "4.3 分部积分法": [
        ('line', 'P43', 'P59'),
        ('line', 'P59', 'P66'),
        ('line', 'P66', 'I'),
        ('line', 'I', 'P43'),
    ],
    "4.4 有理函数的积分": [
        ('line', 'P59', 'P44'),
        ('line', 'P44', 'J'),
        ('line', 'J', 'P66'),
        ('line', 'P66', 'P59'),
    ],
    # 第五章 定积分
    "5.1 定积分概念与性质": [
        ('arc', r1, np.pi / 6, np.pi / 3, h1),
        ('line', 'P17', 'P510'),
        ('line', 'P510', 'P44'),
        ('line', 'P44', 'D'),
    ],
    "5.2 微积分基本公式": [
        ('arc', r1, np.pi / 3, np.pi / 2, h1),
        ('line', 'E', 'P45'),
        ('line', 'P45', 'P510'),
        ('line', 'P510', 'P17'),
    ],
    "5.3 定积分换元与分部积分": [
        ('line', 'P44', 'P510'),
        ('line', 'P510', 'P67'),
        ('line', 'P67', 'J'),
        ('line', 'J', 'P44'),
    ],
    "5.4 反常积分": [
        ('line', 'P510', 'P45'),
        ('line', 'P45', 'K'),
        ('line', 'K', 'P67'),
        ('line', 'P67', 'P510'),
    ],
    # 第六章 定积分的应用
    "6.1 定积分的元素法": [
        ('arc', r1, np.pi / 2, 2 * np.pi / 3, h1),
        ('line', 'P18', 'P512'),
        ('line', 'P512', 'P45'),
        ('line', 'P45', 'E'),
    ],
    "6.2 几何学上的应用": [
        ('line', 'P45', 'P46'),
        ('line', 'P46', 'L'),
        ('line', 'L', 'K'),
        ('line', 'K', 'P45'),
    ],
    "6.3 物理学上的应用": [
        ('arc', r1, 2 * np.pi / 3, 5 * np.pi / 6, h1),
        ('line', 'F', 'P46'),
        ('line', 'P46', 'P512'),
        ('line', 'P512', 'P18'),
    ],
    # 第七章 常微分方程
    "7.1 微分方程基本概念": [
        ('arc', r1, 5 * np.pi / 6, 5 * np.pi / 6 + np.pi / 9, h1),
        ('line', 'P19', 'P515'),
        ('line', 'P515', 'P46'),
        ('line', 'P46', 'F'),
    ],
    "7.2 可分离变量的微分方程": [
        ('arc', r1, 5 * np.pi / 6 + np.pi / 9, 5 * np.pi / 6 + 2 * np.pi / 9, h1),
        ('line', 'P110', 'P517'),
        ('line', 'P517', 'P515'),
        ('line', 'P515', 'P19'),
    ],
    "7.3 齐次方程": [
        ('arc', r1, 5 * np.pi / 6 + 2 * np.pi / 9, 7 * np.pi / 6, h1),
        ('line', 'A', 'P41'),
        ('line', 'P41', 'P517'),
        ('line', 'P517', 'P110'),
    ],
    "7.4 一阶线性微分方程": [
        ('line', 'P46', 'P514'),
        ('line', 'P514', 'P610'),
        ('line', 'P610', 'L'),
        ('line', 'L', 'P46'),
    ],
    "7.5 可降阶的高阶微分方程": [
        ('line', 'P514', 'P516'),
        ('line', 'P516', 'P611'),
        ('line', 'P611', 'P610'),
        ('line', 'P610', 'P514'),
    ],
    "7.6 高阶线性微分方程": [
        ('line', 'P516', 'P518'),
        ('line', 'P518', 'P612'),
        ('line', 'P612', 'P611'),
        ('line', 'P611', 'P516'),
    ],
    "7.7-7.8 常系数线性微分方程": [
        ('line', 'P518', 'P41'),
        ('line', 'P41', 'G'),
        ('line', 'G', 'P612'),
        ('line', 'P612', 'P518'),
    ],
}
# ===================== 绘制图形 =====================
fig = plt.figure(figsize=(14, 14), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
# 绘制基础球面
theta, phi = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
x_sphere = R * np.sin(phi) * np.cos(theta)
y_sphere = R * np.sin(phi) * np.sin(theta)
z_sphere = R * np.cos(phi)
ax.plot_surface(x_sphere, y_sphere, z_sphere,
                color='lightblue', alpha=alpha_sphere,
                rstride=5, cstride=5, edgecolor='none')
# 使用渐变色区分不同章节
colors = plt.cm.hsv(np.linspace(0, 1, len(sections)))
# 绘制所有章节边界曲线
for idx, (section_name, curves) in enumerate(sections.items()):
    current_color = colors[idx]
    for curve in curves:
        if curve[0] == 'arc':
            _, r, t_start, t_end, z_plane = curve
            x, y, z = generate_arc(r, t_start, t_end, z_plane, R)
            ax.plot(x, y, z, color=current_color, linewidth=line_width)
        elif curve[0] == 'arc_rev':
            _, r, t_start, t_end, z_plane = curve
            x, y, z = generate_arc(r, t_end, t_start, z_plane, R)
            ax.plot(x, y, z, color=current_color, linewidth=line_width)
        elif curve[0] == 'line':
            _, p1_name, p2_name = curve
            p1 = all_points[p1_name]
            p2 = all_points[p2_name]
            x, y, z = generate_line(p1, p2, R)
            ax.plot(x, y, z, color=current_color, linewidth=line_width)

# ===================== 新增：端点变量名标注（完全不改动原有逻辑） =====================
# 1. 收集所有线段的端点名称，集合自动去重
label_point_names = set()
for curves in sections.values():
    for curve in curves:
        if curve[0] == 'line':
            label_point_names.add(curve[1])
            label_point_names.add(curve[2])

# 2. 逐个投影到球面并标注变量名
for point_name in label_point_names:
    x_raw, y_raw, z_raw = all_points[point_name]
    x_sp, y_sp, z_sp = project_to_sphere(x_raw, y_raw, z_raw, R)
    # 沿球面径向向外偏移3%，避免与线条重叠
    ax.text(
        x_sp * 1.03, y_sp * 1.03, z_sp * 1.03,
        point_name,
        color='black', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        zorder=10
    )

# 隐藏所有坐标轴和刻度（严格按照要求）
ax.set_axis_off()
# 设置球面范围确保完整显示
ax.set_xlim(-R * 1.05, R * 1.05)
ax.set_ylim(-R * 1.05, R * 1.05)
ax.set_zlim(-R * 1.05, R * 1.05)
# 设置最佳观察视角
ax.view_init(elev=25, azim=45)
# 调整布局消除空白
plt.tight_layout()
# 显示图形
plt.show()