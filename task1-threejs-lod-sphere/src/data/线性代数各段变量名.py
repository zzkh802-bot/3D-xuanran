import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 全局参数（严格按照题目要求）
R = 10  # 球面半径
h1 = 6  # 上投影平面高度
h2 = 8  # 下投影平面高度
r = 7  # 基础圆弧半径
N_POINTS = 100  # 每个线段/圆弧的等分数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 颜色配置（不同章节使用不同颜色，半透明球面）
SPHERE_COLOR = '#f0f0f0'
SPHERE_ALPHA = 0.4  # 40%透明度
CHAPTER_COLORS = {
    '新手村': '#ff4444',
    '第一章': '#00cc99',
    '第二章': '#3366ff',
    '第三章': '#ff9933',
    '第四章': '#9966ff',
    '第五章': '#ff66cc'
}

# 预计算所有常数和关键点坐标
r1 = (np.sqrt(5) - 1) / 2 * r  # 新手村主圆半径
center_O = (np.sqrt(2) / 2 * r, -np.sqrt(2) / 2 * r, h1)  # 新手村主圆圆心O
center_O1 = (np.sqrt(2) / 4 * r, -np.sqrt(2) / 4 * r, h1)  # 圆①圆心O1
center_O2 = (-np.sqrt(2) / 4 * r, np.sqrt(2) / 4 * r, h1)  # 小圆O2圆心
center_big = (0, 0, h1)  # 半径r的大圆圆心

# 下平面关键点坐标（z=-h2）
sqrt_R2_h22 = np.sqrt(R ** 2 - h2 ** 2)
F = (-np.sqrt(3) / 2 * sqrt_R2_h22, -1 / 2 * sqrt_R2_h22, -h2)
G = (np.sqrt(3) / 2 * sqrt_R2_h22, -1 / 2 * sqrt_R2_h22, -h2)
H = (0, sqrt_R2_h22, -h2)
P51 = (-np.sqrt(3) / 6 * sqrt_R2_h22, -1 / 2 * sqrt_R2_h22, -h2)
P52 = (np.sqrt(3) / 6 * sqrt_R2_h22, -1 / 2 * sqrt_R2_h22, -h2)

# 上平面关键点坐标（z=h1）
A = (center_O1[0] + r / 2 * np.cos(-np.pi / 4),
     center_O1[1] + r / 2 * np.sin(-np.pi / 4), h1)
B = (center_O2[0] + r / 2 * np.cos(-np.pi / 4 + np.pi),
     center_O2[1] + r / 2 * np.sin(-np.pi / 4 + np.pi), h1)
C = (r * np.cos(-np.pi / 6), r * np.sin(-np.pi / 6), h1)
D = (r * np.cos(np.pi / 2), r * np.sin(np.pi / 2), h1)
E = (r * np.cos(np.pi / 6 - np.pi), r * np.sin(np.pi / 6 - np.pi), h1)
ad1= (r * np.cos(2*np.pi / 3), r * np.sin(2*np.pi / 3), h1)
ad2= (r * np.cos(5*np.pi / 6), r * np.sin(5*np.pi / 6), h1)
ad3= (r * np.cos(np.pi ), r * np.sin(np.pi ), h1)

# 中点和分点计算
P42 = ((C[0] + G[0]) / 2, (C[1] + G[1]) / 2, (C[2] + G[2]) / 2)
P43 = ((D[0] + H[0]) / 2, (D[1] + H[1]) / 2, (D[2] + H[2]) / 2)
P44 = ((E[0] + F[0]) / 2, (E[1] + F[1]) / 2, (E[2] + F[2]) / 2)
P53 = ((G[0] + H[0]) / 2, (G[1] + H[1]) / 2, (G[2] + H[2]) / 2)
IP54 = ((P42[0] + P43[0]) / 2, (P42[1] + P43[1]) / 2, (P42[2] + P43[2]) / 2)
P55 = ((2 * H[0] + F[0]) / 3, (2 * H[1] + F[1]) / 3, (2 * H[2] + F[2]) / 3)  # HF靠近H的三等分点
P56 = ((H[0] + 2 * F[0]) / 3, (H[1] + 2 * F[1]) / 3, (H[2] + 2 * F[2]) / 3)  # HF靠近F的三等分点
P57 = ((3 * P43[0] + P44[0]) / 4, (3 * P43[1] + P44[1]) / 4, (3 * P43[2] + P44[2]) / 4)  # P43P44靠近P43的四等分点
P58 = ((P43[0] + 3 * P44[0]) / 4, (P43[1] + 3 * P44[1]) / 4, (P43[2] + 3 * P44[2]) / 4)  # P43P44靠近P44的四等分点
IP59 = ((P43[0] + P44[0]) / 2, (P43[1] + P44[1]) / 2, (P43[2] + P44[2]) / 2)  # P43P44中点
P510= ((2 * P43[0] + P44[0]) / 3, (2 * P43[1] + P44[1]) / 3, (2 * P43[2] + P44[2]) / 3)  # P43,P44靠近P43的三等分点
P511= (( P43[0] + 2 *P44[0]) / 3, ( P43[1] +2 * P44[1]) / 3, ( P43[2] + 2 *P44[2]) / 3)  # P43,P44靠近P44的三等分点

def create_sphere(radius=R, resolution=100):
    """创建半透明球面网格"""
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z

def create_arc(center, radius, t_start, t_end, n_points=N_POINTS):
    """创建圆弧段（严格按照参数方程）"""
    t = np.linspace(t_start, t_end, n_points)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z = np.full_like(x, center[2])
    return np.column_stack((x, y, z))

def create_line(p1, p2, n_points=N_POINTS):
    """创建线段（严格按照线性插值公式）"""
    t = np.linspace(0, 1, n_points)
    x = p1[0] + t * (p2[0] - p1[0])
    y = p1[1] + t * (p2[1] - p1[1])
    z = p1[2] + t * (p2[2] - p1[2])
    return np.column_stack((x, y, z))

def project_to_sphere(points, radius=R):
    """严格按照文档公式投影到球面"""
    norms = np.linalg.norm(points, axis=1)
    scale = radius / norms[:, np.newaxis]
    return points * scale

def create_all_arcs():
    """创建文档中定义的所有圆弧段（键名已100%验证）"""
    arcs = {}
    # 新手村主圆弧段（圆心O，半径r1）
    t1 = -0.0915
    t2 = 7 * np.pi / 20
    t3 = np.arccos((np.sqrt(2) * (np.sqrt(5) - 3) - 2 * np.sqrt(2 * (np.sqrt(5) - 2))) / (2 * (np.sqrt(5) - 1)))
    t4 = 23 * np.pi / 20

    arcs['P41_P31'] = create_arc(center_O, r1, t1, t2)
    arcs['P31_tp'] = create_arc(center_O, r1, t2, np.pi / 2)
    arcs['P31_P11'] = create_arc(center_O, r1, t2, t3)
    arcs['tp_P11'] = create_arc(center_O, r1, np.pi / 2, t3)
    arcs['P11_P21'] = create_arc(center_O, r1, t3, t4)
    arcs['P21_P41'] = create_arc(center_O, r1, t4, 2 * np.pi + t1)

    # 圆O1的圆弧段（圆心O1，半径r/2）
    arcs['P11_A'] = create_arc(center_O1, r / 2, -np.pi / 4 - np.arccos(np.sqrt(5) - 2), -np.pi / 4)
    arcs['P12_P11'] = create_arc(center_O1, r / 2, -np.pi / 4 - 4 * np.pi / 5, -np.pi / 4 - np.arccos(np.sqrt(5) - 2))
    arcs['center_big_P12'] = create_arc(center_O1, r / 2, -5 * np.pi / 4, -np.pi / 4 - 4 * np.pi / 5)

    # 圆O2的圆弧段（圆心O2，半径r/2）
    arcs['center_big_P13'] = create_arc(center_O2, r / 2, -np.pi / 4, -np.pi / 4 + np.pi / 5)
    arcs['P13_P14'] = create_arc(center_O2, r / 2, -np.pi / 4 + np.pi / 5, -np.pi / 4 + 3 * np.pi / 5)
    arcs['P13_B'] = create_arc(center_O2, r / 2, -np.pi / 4 + np.pi / 5, -np.pi / 4 + np.pi)
    arcs['P14_B'] = create_arc(center_O2, r / 2, -np.pi / 4 + 3 * np.pi / 5, -np.pi / 4 + np.pi)
    arcs['P32_D'] = create_arc(center_big, r, -np.pi / 4 + 2 * np.pi / 5, np.pi / 2)
    arcs['D_P34'] = create_arc(center_big, r, np.pi / 2, -np.pi / 4 + 4 * np.pi / 5)
    arcs['center_big_P11'] = create_arc(center_O1, r / 2, -np.pi / 4 - np.pi, -np.pi / 4 - np.arccos(np.sqrt(5) - 2))

    # 大圆（圆心(0,0,h1)，半径r）的圆弧段
    arcs['A_C'] = create_arc(center_big, r, -np.pi / 4, -np.pi / 6)
    arcs['C_P31'] = create_arc(center_big, r, -np.pi / 6, -np.pi / 4 + np.pi / 5)
    arcs['P31_P32'] = create_arc(center_big, r, -np.pi / 4 + np.pi / 5, -np.pi / 4 + 2 * np.pi / 5)
    arcs['P32_P33'] = create_arc(center_big, r, -np.pi / 4 + 2 * np.pi / 5, -np.pi / 4 + 3 * np.pi / 5)
    arcs['P33_P34'] = create_arc(center_big, r, -np.pi / 4 + 3 * np.pi / 5, -np.pi / 4 + 4 * np.pi / 5)
    arcs['P33_B'] = create_arc(center_big, r, -np.pi / 4 + 3 * np.pi / 5, -np.pi / 4 + np.pi)
    arcs['P34_B'] = create_arc(center_big, r, -np.pi / 4 + 4 * np.pi / 5, -np.pi / 4 + np.pi)
    arcs['B_P24'] = create_arc(center_big, r, -np.pi / 4 - np.pi, -np.pi / 4 - 4 * np.pi / 5)
    arcs['P24_P23'] = create_arc(center_big, r, -np.pi / 4 - 4 * np.pi / 5, -np.pi / 4 - 3 * np.pi / 5)
    arcs['P23_P22'] = create_arc(center_big, r, -np.pi / 4 - 3 * np.pi / 5, -np.pi / 4 - 2 * np.pi / 5)
    arcs['P22_P21'] = create_arc(center_big, r, -np.pi / 4 - 2 * np.pi / 5, -np.pi / 4 - np.pi / 5)
    arcs['P21_A'] = create_arc(center_big, r, -np.pi / 4 - np.pi / 5, -np.pi / 4)
    arcs['A_P31'] = create_arc(center_big, r, -np.pi / 4, -np.pi / 4 + np.pi / 5)
    arcs['P21_C'] = create_arc(center_big, r, -np.pi / 4 - np.pi / 5, -np.pi / 6)
    arcs['C_P31'] = create_arc(center_big, r, -np.pi / 6, -np.pi / 4 + np.pi / 5)
    arcs['P24_E'] = create_arc(center_big, r, -np.pi / 4 - 4 * np.pi / 5, -np.pi + np.pi / 6)
    arcs['E_P22'] = create_arc(center_big, r, -np.pi + np.pi / 6, -np.pi / 4 - 2 * np.pi / 5)
    arcs['D_ad1']= create_arc(center_big, r, np.pi / 2,2 * np.pi / 3)
    arcs['ad1_ad2'] = create_arc(center_big, r, 2*np.pi / 3, 5 * np.pi / 6)
    arcs['ad2_ad3'] = create_arc(center_big, r, 5 * np.pi / 6,  np.pi )
    arcs['ad3_E'] = create_arc(center_big, r,  np.pi , 7 * np.pi / 6)

    return arcs

def create_all_regions(arcs):
    """创建文档中定义的所有区域边界（所有键名已100%验证）"""
    regions = {}
    P41 = arcs['P41_P31'][0]
    P31 = arcs['P31_tp'][0]
    P11 = arcs['P11_P21'][0]
    P21 = arcs['P21_P41'][0]
    P12 = arcs['P12_P11'][0]
    P13 = arcs['P13_P14'][0]
    P14 = arcs['P14_B'][0]
    P22 = arcs['P22_P21'][0]
    P23 = arcs['P23_P22'][0]
    P24 = arcs['P24_P23'][0]
    P32 = arcs['P32_P33'][0]
    P33 = arcs['P33_P34'][0]
    P34 = arcs['P34_B'][0]
    tp = arcs['tp_P11'][0]

    # 新手村区域
    regions['新手村'] = [
        arcs['P41_P31'], arcs['P31_P11'], arcs['P11_P21'], arcs['P21_P41']
    ]
    # 第一章 行列式
    regions['0.1 二阶三阶行列式'] = [
        arcs['P11_A'][::-1], arcs['P11_P21'], arcs['P21_A']
    ]
    regions['1.2-1.3 全排列与n阶行列式'] = [
        arcs['P12_P11'], arcs['P11_P21'], arcs['P22_P21'][::-1], create_line(P12, P22)[::-1]
    ]
    regions['1.4 行列式的性质'] = [
        arcs['center_big_P13'][::-1], arcs['center_big_P12'], create_line(P12, P22),
        arcs['P23_P22'][::-1], create_line(P23, P13)
    ]
    regions['1.5 按行列展开'] = [
        arcs['P13_P14'][::-1], create_line(P13, P23), arcs['P24_P23'][::-1], create_line(P24, P14)
    ]
    regions['1.6 特殊行列式'] = [
        arcs['P14_B'][::-1], create_line(P14, P24), arcs['B_P24'][::-1]
    ]
    # 第二章 矩阵及其运算
    regions['0.2 二阶三阶矩阵'] = [
        arcs['P11_A'], arcs['A_P31'], arcs['P31_P11']
    ]
    regions['2.1 线性方程组和矩阵'] = [
        create_line(tp, center_big), arcs['center_big_P11'], arcs['tp_P11'][::-1]
    ]
    regions['2.2 矩阵的运算'] = [
        create_line(center_big, tp), arcs['P31_tp'][::-1], arcs['P31_P32'], create_line(P32, center_big)
    ]
    regions['2.3 逆矩阵'] = [
        arcs['P32_P33'], create_line(P33, P13), arcs['center_big_P13'][::-1],
        create_line(center_big, P32)
    ]
    regions['2.5 分块矩阵'] = [
        arcs['P33_B'][::-1], create_line(P33, P13), arcs['P13_B']
    ]
    # 第三章 矩阵的初等变换与线性方程组
    regions['0.3 二元三元方程组'] = [
        arcs['P21_C'][::-1], arcs['P21_P41'], create_line(C, P41)[::-1]
    ]
    regions['3.1 矩阵的初等变换'] = [
        arcs['P22_P21'], create_line(P21, P52), create_line(P52, P51), create_line(P51, P22)
    ]
    regions['3.2 矩阵的秩'] = [
        arcs['E_P22'], create_line(P22, P51), create_line(P51, F), create_line(F, E)
    ]
    regions['3.3 线性方程组的解'] = [
        create_line(P21, P52), create_line(P52, G), create_line(G, P42),
        create_line(P42, P41), arcs['P21_P41'][::-1]
    ]
    # 第四章 向量组的线性相关性
    regions['0.4 二维三维向量'] = [
        create_line(C, P41), arcs['P41_P31'], arcs['C_P31'][::-1]
    ]
    regions['4.1 向量组及其线性组合'] = [
        create_line(P42, IP54), create_line(IP54, P32), arcs['P31_P32'][::-1],
        arcs['P41_P31'][::-1], create_line(P41, P42)
    ]
    regions['4.2 向量组的线性相关性'] = [
        create_line(P42, G), create_line(G, P53), create_line(P53, IP54), create_line(IP54, P42)
    ]
    regions['4.3 向量组的秩'] = [
        create_line(P32, IP54), create_line(IP54, P43), create_line(P43, D), arcs['P32_D'][::-1]
    ]
    regions['4.4 线性方程组的解的结构'] = [
        create_line(IP54, P43), create_line(P43, H), create_line(H, P53), create_line(P53, IP54)
    ]
    regions['4.5 向量空间'] = [
        create_line(F, G), create_line(G, H), create_line(H, F)
    ]
    # 第五章 相似矩阵及二次型
    regions['5.1 内积长度正交性'] = [
         arcs['D_ad1'], create_line(ad1, P57), create_line(P57, P43), create_line(P43, D)
    ]
    regions['5.2 特征值与特征向量'] = [
        arcs['ad1_ad2'], create_line(ad2, IP59), create_line(IP59, P57), create_line(P57, ad1)
    ]
    regions['5.3 相似矩阵'] = [
        arcs['ad2_ad3'], create_line(ad3, P58), create_line(P58, IP59), create_line(IP59, ad2)
    ]
    regions['5.4 对称矩阵的对角化'] = [
        arcs['ad3_E'], create_line(E, P44), create_line(P44, P58), create_line(P58, ad3)
    ]
    regions['5.5 二次型及其标准化'] = [
        create_line(P44, F), create_line(F, P56), create_line(P56, P511), create_line(P511, P44)
    ]
    regions['5.6 配方法化标准型'] = [
        create_line(P511, P56), create_line(P56, P55), create_line(P55, P510), create_line(P510, P511)
    ]
    regions['5.7 正定二次型'] = [
        create_line(P510, P55), create_line(P55, H), create_line(H, P43), create_line(P43, P510)
    ]
    return regions

# ========== 修复后的端点收集&去重函数 ==========
def collect_all_label_points(arcs):
    """收集所有端点，纯元组去重，彻底解决数组判断报错"""
    # 所有待标注端点名称 + 原始坐标
    base_points = {
        'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
        'F': F, 'G': G, 'H': H, 'ad1': ad1, 'ad2': ad2, 'ad3': ad3,
        'P41': arcs['P41_P31'][0], 'P31': arcs['P31_tp'][0], 'P11': arcs['P11_A'][0],
        'P21': arcs['P21_P41'][0], 'P12': arcs['P12_P11'][0], 'P13': arcs['P13_P14'][0],
        'P14': arcs['P14_B'][0], 'P22': arcs['P22_P21'][0], 'P23': arcs['P23_P22'][0],
        'P24': arcs['P24_P23'][0], 'P32': arcs['P32_P33'][0], 'P33': arcs['P33_P34'][0],
        'P34': arcs['P34_B'][0], 'tp': arcs['tp_P11'][0], 'P42': P42, 'P43': P43,
        'P44': P44, 'P51': P51, 'P52': P52, 'P53': P53, 'IP54': IP54,
        'P55': P55, 'P56': P56, 'P57': P57, 'P58': P58, 'IP59': IP59,
        'P510': P510, 'P511': P511,'center_big': center_big
        # 如需标注圆心，解除下面注释即可
        # 'center_O': center_O, 'center_O1': center_O1,
        # 'center_O2': center_O2,
    }

    unique_points = {}
    seen_coords = set()  # 用集合存储已出现坐标元组，用于去重

    for name, coord in base_points.items():
        # 统一转为numpy数组 -> 四舍五入 -> 纯Python元组
        arr = np.array(coord)
        coord_tuple = tuple(round(x, 4) for x in arr)
        # 未出现过则加入
        if coord_tuple not in seen_coords:
            seen_coords.add(coord_tuple)
            unique_points[name] = coord

    return unique_points

def plot_spherical_regions():
    """绘制球面、区域边界 + 不重复端点标注"""
    fig = plt.figure(figsize=(14, 12), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制半透明球面
    x_sphere, y_sphere, z_sphere = create_sphere()
    ax.plot_surface(x_sphere, y_sphere, z_sphere,
                    color=SPHERE_COLOR, alpha=SPHERE_ALPHA,
                    rstride=5, cstride=5, linewidth=0)

    arcs = create_all_arcs()
    regions = create_all_regions(arcs)

    # 绘制所有区域边界
    for region_name, boundaries in regions.items():
        if '新手村' in region_name:
            color = CHAPTER_COLORS['新手村']
        elif region_name.startswith('1'):
            color = CHAPTER_COLORS['第一章']
        elif region_name.startswith('2'):
            color = CHAPTER_COLORS['第二章']
        elif region_name.startswith('3'):
            color = CHAPTER_COLORS['第三章']
        elif region_name.startswith('4'):
            color = CHAPTER_COLORS['第四章']
        elif region_name.startswith('5'):
            color = CHAPTER_COLORS['第五章']
        else:
            color = 'black'

        for boundary in boundaries:
            projected = project_to_sphere(boundary)
            ax.plot(projected[:, 0], projected[:, 1], projected[:, 2],
                    color=color, linewidth=2, label=region_name)

    # 绘制端点标注
    label_points = collect_all_label_points(arcs)
    for point_name, coord in label_points.items():
        point_array = np.array([coord])
        projected_point = project_to_sphere(point_array)[0]
        # 文本标注样式：白底半透明、加粗、置顶
        ax.text(
            projected_point[0], projected_point[1], projected_point[2],
            point_name,
            color='black', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            zorder=10
        )

    # 关闭坐标轴、网格
    ax.set_axis_off()
    ax.grid(False)
    # 坐标范围
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    # 视角
    ax.view_init(elev=25, azim=50)

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(),
              loc='upper left', bbox_to_anchor=(1.08, 1), fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_spherical_regions()