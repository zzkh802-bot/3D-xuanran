import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label as ndi_label, binary_erosion
from scipy.spatial import KDTree, ConvexHull
from matplotlib.path import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORT_DIR = os.path.join(SCRIPT_DIR, '棋盘')
if SUPPORT_DIR not in sys.path:
    sys.path.insert(0, SUPPORT_DIR)

from saddle_and_mark_from_m import (
    imread_gray, compute_gradients, create_correlation_patch,
    filter_image_with_templates_fft, non_maximum_suppression_fast,
    find_modes_mean_shift, edge_orientations_fast, refine_corners_vectorized,
    corner_correlation_score, score_corners_batch, measure_blur_level,
    merge_close_points, adjust_orientation_handedness,
    compute_homography, get_little_four_P_py, grow_chessboard_region,
    get_mark_cord as get_mark_cord_m
)


# ===================== 新增：共线性计算工具函数 =====================
def calculate_collinearity_residual(points):
    """
    计算点集的共线性残差（越小越共线）
    :param points: (N,2) 像素坐标点集
    :return: 平均残差（越小共线性越好）
    """
    if len(points) < 2:
        return 0.0
    # 最小二乘拟合直线 y = kx + b（处理垂直线特殊情况）
    x = points[:, 0]
    y = points[:, 1]
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 计算斜率k和截距b
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator < 1e-12:  # 垂直线
        k = np.inf
        b = x_mean
        # 残差为x到垂直线的距离
        residuals = np.abs(x - b)
    else:
        k = numerator / denominator
        b = y_mean - k * x_mean
        # 残差为点到直线的垂直距离
        residuals = np.abs(k * x - y + b) / np.sqrt(k ** 2 + 1)

    return np.mean(residuals)


def evaluate_collinearity_consistency(existing_pts, new_pt, grid_dir):
    """
    评估新增点与现有同方向网格点的共线性一致性
    :param existing_pts: (N,2) 现有同方向网格点
    :param new_pt: (2,) 新增点坐标
    :param grid_dir: 网格方向（0=水平/左右, 1=垂直/上下）
    :return: 一致性评分（越高越一致，范围[0,1]）
    """
    if len(existing_pts) < 2:
        return 1.0  # 现有点数不足，默认一致

    # 拼接现有点和新增点
    test_pts = np.vstack([existing_pts, new_pt])

    # 按网格方向筛选：以新增点的坐标作为参考线
    if grid_dir == 0:  # 水平/左右方向（同y坐标）
        ref_y = new_pt[1]
        mask = np.isclose(test_pts[:, 1], ref_y, atol=1.5)
        collinear_pts = test_pts[mask]
    else:  # 垂直/上下方向（同x坐标）
        ref_x = new_pt[0]
        mask = np.isclose(test_pts[:, 0], ref_x, atol=1.5)
        collinear_pts = test_pts[mask]

    # 计算共线性残差
    residual = calculate_collinearity_residual(collinear_pts)
    # 转换为评分（残差越小评分越高）
    max_residual = 10.0  # 最大容忍残差（像素），放宽以适应镜头畸变和透视
    score = max(0.0, 1.0 - residual / max_residual)

    return score


def evaluate_local_grid_alignment(new_local_pt, anchor_local_pt, grid_dir, ref_len=0):
    """
    Score whether a new point stays on the expected local grid line.
    移除了未使用的 existing_local_pts 参数，直接基于锚点评估垂直偏移与步长。
    """
    perpendicular_axis = 1 if grid_dir == 0 else 0
    travel_axis = 0 if grid_dir == 0 else 1
    tolerance = max(3.5, ref_len * 0.28) if ref_len > 0 else 5.0  # 放宽垂直容差

    perp_error = abs(new_local_pt[perpendicular_axis] - anchor_local_pt[perpendicular_axis])
    perp_score = np.exp(-0.5 * (perp_error / (tolerance + 1e-12)) ** 2)

    if ref_len <= 0:
        return float(perp_score)

    step = abs(new_local_pt[travel_axis] - anchor_local_pt[travel_axis])
    step_error = abs(step - ref_len)
    step_score = np.exp(-0.5 * (step_error / (ref_len * 0.38 + 1e-12)) ** 2)  # 放宽步长容差
    return float(0.72 * perp_score + 0.28 * step_score)


def normalize_image_for_corners(img_gray):
    """Percentile normalization reduces background drift before corner detection."""
    lo, hi = np.percentile(img_gray, [1.0, 99.0])
    if hi - lo < 1e-8:
        return (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
    return np.clip((img_gray - lo) / (hi - lo), 0, 1)


# ===================== 原有函数：新增共线性校验逻辑 =====================
def build_local_frame(center_pt, v1, v2):
    """构建以center_pt为原点、v1为x轴、正交化v2为y轴的局部坐标系"""
    v1_u = v1 / (np.linalg.norm(v1) + 1e-12)
    v2_proj = v2 - np.dot(v2, v1_u) * v1_u
    v2_u = v2_proj / (np.linalg.norm(v2_proj) + 1e-12)
    R = np.array([v1_u, v2_u])  # 2x2
    return R, center_pt


def transform_points(points, R, center_pt):
    """将点集变换到局部坐标系"""
    return (points - center_pt) @ R.T


def grow_neighbor(xc0, used_mask, center_idx, direction_vec, ref_len=0,
                  original_pts=None, grid_dir=0):
    """
    在局部坐标系 xc0 中，从 center_idx 沿方向 direction_vec 寻找下一个点。
    新增：结合像素共线性一致性评分筛选最优点
    :param original_pts: (N,2) 原始像素坐标点集（用于共线性计算，已移除此逻辑）
    :param grid_dir: 网格方向（0=水平/左右, 1=垂直/上下）
    :return: 候选索引，若找不到返回 -1。
    """
    center = xc0[center_idx]
    vecs = xc0 - center
    dists = np.linalg.norm(vecs, axis=1)
    # 投影（带符号）
    proj = vecs @ direction_vec
    cos_angle = proj / (dists + 1e-12)
    # 方向约束：0.80=≈37°容忍度，适应角落透视畸变（参考yanzheng2026_4_29/5_12）
    mask_dir = (cos_angle > 0.80) & (proj > 0)
    # 排除自身和已使用点
    mask_avail = ~used_mask
    mask_avail[center_idx] = False
    mask = mask_dir & mask_avail
    if ref_len > 0:
        ratio = dists / (ref_len + 1e-12)
        mask &= (ratio > 0.50) & (ratio < 1.60)  # 放宽间距比例，适应透视压缩/拉伸
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        return -1

    # 1. 基础评分（夹角+距离）
    if ref_len > 1e-6:
        dist_scores = np.clip(
            1.0 - np.abs(dists[candidates] - ref_len) / (ref_len * 0.35 + 1e-12),
            0.0, 1.0
        )
    else:
        scale = np.percentile(dists[candidates], 25) + 1e-12
        dist_scores = 1.0 / (1.0 + dists[candidates] / scale)
    base_scores = np.clip(cos_angle[candidates], 0.0, 1.0) * dist_scores

    # 2. 局部网格对齐评分（替代原先的死代码和像素共线性逻辑）
    collinear_scores = np.ones_like(base_scores)
    for i, cand_idx in enumerate(candidates):
        collinear_scores[i] = evaluate_local_grid_alignment(
            xc0[cand_idx], center, grid_dir, ref_len
        )

    # 3. 综合评分（基础评分*共线性评分）
    final_scores = base_scores * collinear_scores
    best = candidates[np.argmax(final_scores)]

    return best


def assign_grid_coords(adj_matrix, start_idx):
    """
    根据邻接矩阵 (Nx4，列序：左、右、上、下，-1 表示无连接) 分配网格坐标。
    返回 (N,2) 数组，若发生冲突则返回 None。
    列顺序：[row, col]
    """
    n = adj_matrix.shape[0]
    coords = np.full((n, 2), np.nan)
    coords[start_idx] = [0, 0]
    queue = [start_idx]
    while queue:
        cur = queue.pop(0)
        r, c = coords[cur]
        # 左 (0) -> c-1
        left = adj_matrix[cur, 0]
        if left != -1:
            nc = np.array([r, c - 1])
            if np.isnan(coords[left, 0]):
                coords[left] = nc
                queue.append(left)
            elif not np.array_equal(coords[left], nc):
                return None
        right = adj_matrix[cur, 1]
        if right != -1:
            nc = np.array([r, c + 1])
            if np.isnan(coords[right, 0]):
                coords[right] = nc
                queue.append(right)
            elif not np.array_equal(coords[right], nc):
                return None
        up = adj_matrix[cur, 2]
        if up != -1:
            nc = np.array([r - 1, c])
            if np.isnan(coords[up, 0]):
                coords[up] = nc
                queue.append(up)
            elif not np.array_equal(coords[up], nc):
                return None
        down = adj_matrix[cur, 3]
        if down != -1:
            nc = np.array([r + 1, c])
            if np.isnan(coords[down, 0]):
                coords[down] = nc
                queue.append(down)
            elif not np.array_equal(coords[down], nc):
                return None
    return coords


def chessboardsgrow_py(points, directions, scores, img_shape, max_boards=3):
    """
    移植自 chessboardsgrow0802.m，返回棋盘格列表。
    新增：每次生长后校验新增点的像素共线性一致性
    directions: Nx4 (v1x,v1y,v2x,v2y)
    scores: 每个鞍点的响应值，用于种子排序。
    """
    N = len(points)
    if N == 0:
        return []
    # 种子按响应值降序
    seed_order = np.argsort(scores)[::-1]
    used_global = set()
    chessboards = []
    for seed_rank, seed_idx in enumerate(seed_order[:300]):
        if len(chessboards) >= max_boards:
            break
        if seed_idx in used_global:
            continue
        v1 = directions[seed_idx, :2]
        v2 = directions[seed_idx, 2:4]
        # 构建局部正交坐标系
        v1_u = v1 / (np.linalg.norm(v1) + 1e-12)
        v2_proj = v2 - np.dot(v2, v1_u) * v1_u
        v2_u = v2_proj / (np.linalg.norm(v2_proj) + 1e-12)
        R = np.array([v1_u, v2_u])  # 行向量：x轴，y轴
        center = points[seed_idx]
        xc0 = (points - center) @ R.T  # Nx2 局部坐标
        # 初始化棋盘格数据结构
        board = []
        glob_to_loc = {}
        board.append(seed_idx)
        glob_to_loc[seed_idx] = 0
        adj = -np.ones((1, 4), dtype=int)  # 左、右、上、下
        grow_queue = [(0, 0), (0, 1), (0, 2), (0, 3)]
        fail = False
        while grow_queue and not fail:
            bid, d = grow_queue.pop(0)
            if adj[bid, d] != -1:
                continue
            gid = board[bid]
            center0 = xc0[gid]
            # 方向向量 + 网格方向（用于共线性计算）
            if d == 0:  # 左
                dir_vec = np.array([-1.0, 0.0])
                grid_dir = 0  # 水平方向
            elif d == 1:  # 右
                dir_vec = np.array([1.0, 0.0])
                grid_dir = 0  # 水平方向
            elif d == 2:  # 上
                dir_vec = np.array([0.0, -1.0])
                grid_dir = 1  # 垂直方向
            else:  # 下
                dir_vec = np.array([0.0, 1.0])
                grid_dir = 1  # 垂直方向
            ref_len = 0
            for nd in range(4):
                nid = adj[bid, nd]
                if nid != -1:
                    ngid = board[nid]
                    ref_len = np.linalg.norm(xc0[ngid] - center0)
                    break
            used_mask = np.zeros(N, dtype=bool)
            for g in board:
                used_mask[g] = True

            # 生长邻居：传入原始点和网格方向，用于共线性筛选
            next_gid = grow_neighbor(
                xc0, used_mask, gid, dir_vec, ref_len,
                original_pts=points, grid_dir=grid_dir
            )
            if next_gid == -1:
                continue

            # 新增：校验新增点与现有网格的共线性一致性
            existing_board_pts = points[board]  # 现有棋盘格点（像素坐标）
            new_pt = points[next_gid]
            collinearity_score = evaluate_collinearity_consistency(
                existing_board_pts, new_pt, grid_dir
            )
            # 共线性一致性阈值（可调整）
            collinearity_thresh = 0.20  # 降低共线性一致性阈值，避免畸变/透视区域误拒
            if collinearity_score < collinearity_thresh:
                print(f"  跳过共线性不一致的点（评分：{collinearity_score:.2f} < {collinearity_thresh}）")
                continue  # 跳过不一致的点

            if next_gid not in glob_to_loc:
                new_bid = len(board)
                board.append(next_gid)
                glob_to_loc[next_gid] = new_bid
                adj = np.vstack([adj, -np.ones((1, 4), dtype=int)])
                for nd in range(4):
                    grow_queue.append((new_bid, nd))
            else:
                new_bid = glob_to_loc[next_gid]
            adj[bid, d] = new_bid
            opp_d = d + 1 if d % 2 == 0 else d - 1
            adj[new_bid, opp_d] = bid
        if fail or len(board) < 9:
            continue
        grid_coords = assign_grid_coords(adj, 0)
        if grid_coords is None:
            continue
        valid = ~np.isnan(grid_coords[:, 0])
        if np.sum(valid) < 4:
            continue
        # 修正世界坐标：列索引对应 x，行索引对应 y
        world_local = np.column_stack([grid_coords[:, 1] * 15.0, grid_coords[:, 0] * 15.0])
        img_local = points[board]
        img_pts = img_local[valid]
        world_pts = world_local[valid]
        # ---------- 手性自动修正 ----------
        if len(img_pts) >= 3:
            p0, p1, p2 = img_pts[:3]
            img_cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            w0, w1, w2 = world_pts[:3]
            world_cross = (w1[0] - w0[0]) * (w2[1] - w0[1]) - (w1[1] - w0[1]) * (w2[0] - w0[0])
            if img_cross * world_cross < 0:
                # 翻转 x 轴以保持手性一致（而不是交换 XY）
                world_pts[:, 0] *= -1.0
        # ---------------------------------
        used_indices = set(board)
        H = compute_homography(world_pts, img_pts)
        if H is not None:
            chessboards.append((img_pts, world_pts, H, used_indices))
            used_global.update(used_indices)
    return chessboards


# ===================== 原有函数（保持不变） =====================
def compute_reprojection_error(H, xy, uv):
    if xy.shape[1] == 2:
        xy_hom = np.hstack([xy, np.ones((xy.shape[0], 1))])
    else:
        xy_hom = xy
    uv_proj_hom = (H @ xy_hom.T).T
    uv_proj = uv_proj_hom[:, :2] / uv_proj_hom[:, 2, np.newaxis]
    errors = np.linalg.norm(uv_proj - uv, axis=1)
    mean_error = np.mean(errors)
    return errors, mean_error


def _unique_world_points(img_pts, world_pts):
    world_rounded = np.round(world_pts, 2)
    _, unique_idx = np.unique(world_rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return img_pts[unique_idx], world_pts[unique_idx]


def _bbox_overlap_ratio(a_pts, b_pts):
    if len(a_pts) == 0 or len(b_pts) == 0:
        return 0.0
    a_min = np.min(a_pts, axis=0)
    a_max = np.max(a_pts, axis=0)
    b_min = np.min(b_pts, axis=0)
    b_max = np.max(b_pts, axis=0)
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(0.0, inter_max - inter_min)
    inter_area = float(inter[0] * inter[1])
    a_area = float(np.prod(np.maximum(0.0, a_max - a_min)))
    b_area = float(np.prod(np.maximum(0.0, b_max - b_min)))
    return inter_area / max(1e-6, min(a_area, b_area))


def _is_duplicate_board(img_pts, used_idx, boards, overlap_thresh=0.68):
    used_idx = set(used_idx)
    for exist_img, _, _, exist_used in boards:
        exist_used = set(exist_used)
        if used_idx and exist_used:
            common = len(used_idx & exist_used)
            if common > 0.45 * min(len(used_idx), len(exist_used)):
                return True
        if _bbox_overlap_ratio(img_pts, exist_img) > overlap_thresh:
            return True
    return False


def legacy_homography_chessboardsgrow_py(points, directions, scores, img_shape, max_boards=3):
    """Fallback growth: seed a small quad, then expand through homography projections."""
    if len(points) == 0:
        return []
    dirs = np.stack([directions[:, :2], directions[:, 2:4]], axis=1)
    seed_order = np.argsort(scores)[::-1]
    boards = []
    used_global = set()
    for seed_idx in seed_order[:min(260, len(seed_order))]:
        seed_idx = int(seed_idx)
        if seed_idx in used_global:
            continue
        try:
            _, quad_idx = get_little_four_P_py(points, dirs, seed_idx, img_shape)
        except Exception:
            continue
        if quad_idx is None:
            continue
        try:
            img_pts, world_pts, H, used_idx = grow_chessboard_region(
                points, dirs, quad_idx, max_iter=120, dist_thresh=12.0
            )
        except Exception:
            continue
        if len(img_pts) < 9:
            continue
        img_pts, world_pts = _unique_world_points(img_pts, world_pts)
        if len(img_pts) < 9:
            continue
        H = compute_homography(world_pts, img_pts)
        if H is None:
            continue
        used_idx = set(used_idx)
        if _is_duplicate_board(img_pts, used_idx, boards):
            continue
        boards.append((img_pts, world_pts, H, used_idx))
        used_global.update(used_idx)
        if len(boards) >= max_boards:
            break
    return boards


def find_chessboard_candidates(points, directions, scores, img_shape, max_boards=3):
    """棋盘格检测：使用4_29的单应投影方法为主，BFS方向生长为辅。
    单应投影方法（grow_chessboard_region）无方向约束，天然处理透视畸变，
    对角落鞍点的连接能力远优于BFS方法。"""
    # 主方法：4_29的单应投影生长（get_little_four_P_py + grow_chessboard_region）
    homo_boards = legacy_homography_chessboardsgrow_py(
        points, directions, scores, img_shape, max_boards=max_boards)
    if homo_boards:
        print(f"  单应生长候选 {len(homo_boards)} 个")
    # 辅方法：BFS方向生长（chessboardsgrow_py）
    bfs_boards = chessboardsgrow_py(
        points, directions, scores, img_shape, max_boards=max_boards)
    if bfs_boards:
        print(f"  BFS生长候选 {len(bfs_boards)} 个")
    # 单应结果优先返回（在后续选择中享有优先级）
    return homo_boards + bfs_boards


def _available_strip_points_mask(points, existing_boards, margin=12.0):
    """Exclude detections already covered by a previously accepted board."""
    available = np.ones(len(points), dtype=bool)
    if not existing_boards:
        return available
    existing_pts = np.vstack([b[0] for b in existing_boards if len(b[0])])
    if len(existing_pts):
        dist, _ = KDTree(existing_pts).query(points, k=1)
        available &= dist > margin
    # Distance-only exclusion is insufficient for a sparse broad board: its
    # undetected interior nodes can be more than 12 px from every accepted
    # node and were rediscovered as a fake two-column sub-board (hewei181).
    for board in existing_boards:
        board_pts = np.asarray(board[0], dtype=float)
        if len(board_pts) < 3:
            continue
        try:
            hull = ConvexHull(board_pts)
            covered = Path(board_pts[hull.vertices]).contains_points(
                points, radius=2.0 * margin)
            available &= ~covered
        except Exception:
            # Collinear boards are already handled by the distance gate.
            pass
    return available


def _is_broad_board_lattice_extension(point, existing_boards,
                                       grid_spacing=15.0, margin_cells=2):
    """Return True when a putative X lies on an extended broad-board grid."""
    p = np.asarray(point, dtype=float)
    for board in existing_boards:
        img_pts = np.asarray(board[0], dtype=float)
        world_pts = np.asarray(board[1], dtype=float)
        if len(img_pts) < 9 or len(world_pts) != len(img_pts):
            continue
        grid = np.round(world_pts / grid_spacing).astype(int)
        if len(np.unique(grid[:, 0])) < 4 or len(np.unique(grid[:, 1])) < 4:
            continue
        # Refit from the accepted point/world pairs.  Candidate completion and
        # world-origin normalization may change those pairs after the stored
        # seed homography was created; using that stale matrix made a genuine
        # outer column miss this guard on hewei207.
        H = compute_homography(world_pts, img_pts)
        if H is None:
            H = board[2] if len(board) > 2 else None
        try:
            H_inv = np.linalg.inv(H)
        except (TypeError, np.linalg.LinAlgError):
            continue
        p_h = H_inv @ np.array([p[0], p[1], 1.0])
        if abs(float(p_h[2])) <= 1e-9:
            continue
        units = p_h[:2] / p_h[2] / grid_spacing
        rounded = np.round(units).astype(int)
        if np.linalg.norm(units - rounded) > 0.32:
            continue
        grid_min = np.min(grid, axis=0) - int(margin_cells)
        grid_max = np.max(grid, axis=0) + int(margin_cells)
        if np.all(rounded >= grid_min) and np.all(rounded <= grid_max):
            return True
    return False


def _is_near_broad_board_image_region(point, existing_boards,
                                       margin_scale=3.0):
    """Reject X candidates on or immediately beside a broad checkerboard."""
    p = np.asarray(point, dtype=float)
    for board in existing_boards:
        img_pts = np.asarray(board[0], dtype=float)
        world_pts = np.asarray(board[1], dtype=float)
        if len(img_pts) < 12 or len(world_pts) != len(img_pts):
            continue
        grid = np.round(world_pts / 15.0).astype(int)
        if len(np.unique(grid[:, 0])) < 4 or len(np.unique(grid[:, 1])) < 4:
            continue
        tree = KDTree(img_pts)
        k = min(3, len(img_pts))
        neighbour_dist, _ = tree.query(img_pts, k=k)
        if neighbour_dist.ndim == 2 and neighbour_dist.shape[1] >= 2:
            spacing = float(np.median(neighbour_dist[:, 1]))
        else:
            spacing = 20.0
        margin = float(np.clip(margin_scale * spacing, 18.0, 60.0))
        if (np.min(img_pts[:, 0]) - margin <= p[0] <=
                np.max(img_pts[:, 0]) + margin and
                np.min(img_pts[:, 1]) - margin <= p[1] <=
                np.max(img_pts[:, 1]) + margin):
            return True
    return False


def _merge_cleaned_broad_board_extensions(boards, grid_spacing=15.0):
    """Merge duplicate mini-boards that are one omitted broad-board border.

    Propagation deliberately searches a full translated board.  Under strong
    perspective it can match only the source's omitted outer column and label
    that run as a new sparse board.  At this late stage the cleaned topology
    is accurate enough to map the run back to the source.  Absorption requires
    coherent support from most of the other board and either one exact adjacent
    border line or at least two already shared lattice cells, so independent
    physical boards cannot be joined merely because their boxes are nearby.
    """
    result = list(boards)
    removed = set()

    for target_idx in range(len(result)):
        if target_idx in removed:
            continue
        target_img, target_world, target_H, target_used, target_qual = result[target_idx]
        target_img = np.asarray(target_img, dtype=float)
        target_world = np.asarray(target_world, dtype=float)
        target_grid = np.round(target_world / grid_spacing).astype(int)
        if (len(target_img) < 12 or
                len(np.unique(target_grid[:, 0])) < 4 or
                len(np.unique(target_grid[:, 1])) < 4):
            continue

        for other_idx in range(len(result)):
            if other_idx == target_idx or other_idx in removed:
                continue
            other_img, _, _, other_used, _ = result[other_idx]
            other_img = np.asarray(other_img, dtype=float)
            if len(other_img) < 3:
                continue

            fitted_H = compute_homography(target_world, target_img)
            try:
                fitted_H_inv = np.linalg.inv(fitted_H)
            except (TypeError, np.linalg.LinAlgError):
                continue

            other_h = np.column_stack([other_img, np.ones(len(other_img))])
            mapped_h = (fitted_H_inv @ other_h.T).T
            finite = np.abs(mapped_h[:, 2]) > 1e-9
            units = np.full((len(other_img), 2), np.nan, dtype=float)
            units[finite] = (mapped_h[finite, :2] /
                             mapped_h[finite, 2, np.newaxis] /
                             grid_spacing)
            snapped = np.zeros((len(other_img), 2), dtype=int)
            snapped[finite] = np.round(units[finite]).astype(int)
            error = np.full(len(other_img), np.inf, dtype=float)
            error[finite] = np.linalg.norm(units[finite] - snapped[finite], axis=1)

            c_min, r_min = np.min(target_grid, axis=0)
            c_max, r_max = np.max(target_grid, axis=0)
            in_one_cell = (
                (snapped[:, 0] >= c_min - 1) &
                (snapped[:, 0] <= c_max + 1) &
                (snapped[:, 1] >= r_min - 1) &
                (snapped[:, 1] <= r_max + 1))
            strict = finite & in_one_cell & (error < 0.34)
            if np.sum(strict) < max(3, int(np.ceil(0.65 * len(other_img)))):
                continue

            occupied = {tuple(cell) for cell in target_grid}
            strict_cells = snapped[strict]
            outside = [tuple(cell) for cell in strict_cells
                       if not (c_min <= cell[0] <= c_max and
                               r_min <= cell[1] <= r_max)]
            boundary = None
            boundary_tests = [
                (0, c_min - 1), (0, c_max + 1),
                (1, r_min - 1), (1, r_max + 1),
            ]
            if outside:
                for axis, value in boundary_tests:
                    if all(cell[axis] == value for cell in outside):
                        boundary = (axis, value)
                        break
                if boundary is None:
                    continue
            else:
                # Once part of a boundary run has already been merged, a
                # duplicate propagation board becomes an in-extent hole fill.
                # Require actual shared cells before absorbing that remainder.
                shared = sum(tuple(cell) in occupied for cell in strict_cells)
                if shared < 2:
                    continue

            permissive = finite & in_one_cell & (error < 0.46)
            if boundary is not None:
                axis, value = boundary
                permissive &= snapped[:, axis] == value

            selected = {}
            for point_idx in np.where(permissive)[0]:
                cell = tuple(snapped[int(point_idx)])
                if cell in occupied:
                    continue
                if (cell not in selected or
                        error[point_idx] < error[selected[cell]]):
                    selected[cell] = int(point_idx)

            if selected:
                add_idx = list(selected.values())
                target_img = np.vstack([target_img, other_img[add_idx]])
                target_world = np.vstack([
                    target_world,
                    snapped[add_idx].astype(float) * grid_spacing,
                ])
                target_img, target_world = _unique_world_points(
                    target_img, target_world)
                target_grid = np.round(target_world / grid_spacing).astype(int)
                merged_H = compute_homography(target_world, target_img)
                if merged_H is not None:
                    target_H = merged_H
                target_qual = max(
                    target_qual,
                    calculate_chessboard_quality(target_img, target_world))
                print(f"  [宽棋盘边界归并] 棋盘#{other_idx + 1} -> "
                      f"棋盘#{target_idx + 1}: +{len(add_idx)}个真实鞍点")

            target_used = set(target_used) | set(other_used)
            result[target_idx] = (
                target_img, target_world, target_H, target_used, target_qual)
            removed.add(other_idx)

    return [board for idx, board in enumerate(result) if idx not in removed]


def _rebuild_broad_board_from_complete_image_rows(
        boards, candidate_pts, grid_spacing=15.0):
    """Recover physical columns skipped by a compressed broad-board model.

    A homography grown from alternating seed columns can describe an 8-column
    lattice even when ten genuine saddles are visible in every image row.  In
    that state the ordinary missing-cell routines cannot help: the two omitted
    physical columns project near half-integer coordinates whose neighbouring
    integer cells are already occupied.  Rebuild only when several rows
    independently contain the same, larger number of aligned real candidates.
    Full rows establish the projective topology; incomplete rows are then
    matched to it.  No synthetic saddle is returned.
    """
    candidates = np.asarray(candidate_pts, dtype=float)
    if len(candidates) < 20:
        return list(boards)

    rebuilt = list(boards)
    for board_idx, board in enumerate(rebuilt):
        img_pts, world_pts, old_H, used_idx, old_qual = board
        img_pts = np.asarray(img_pts, dtype=float)
        world_pts = np.asarray(world_pts, dtype=float)
        if len(img_pts) < 40 or len(world_pts) != len(img_pts):
            continue
        grid = np.round(world_pts / grid_spacing).astype(int)
        axis_counts = [len(np.unique(grid[:, axis])) for axis in (0, 1)]
        if min(axis_counts) < 4 or max(axis_counts) < 8:
            continue

        row_axis = int(np.argmax(axis_counts))
        col_axis = 1 - row_axis
        current_cols = axis_counts[col_axis]
        row_values = np.arange(np.min(grid[:, row_axis]),
                               np.max(grid[:, row_axis]) + 1)
        if len(row_values) < 6:
            continue

        fitted_H = compute_homography(world_pts, img_pts)
        try:
            fitted_H_inv = np.linalg.inv(fitted_H)
        except (TypeError, np.linalg.LinAlgError):
            continue
        cand_h = np.column_stack([candidates, np.ones(len(candidates))])
        mapped_h = (fitted_H_inv @ cand_h.T).T
        finite = np.abs(mapped_h[:, 2]) > 1e-9
        units = np.full((len(candidates), 2), np.nan, dtype=float)
        units[finite] = (mapped_h[finite, :2] /
                         mapped_h[finite, 2, np.newaxis] /
                         grid_spacing)
        rounded_rows = np.zeros(len(candidates), dtype=int)
        rounded_rows[finite] = np.round(
            units[finite, row_axis]).astype(int)
        row_error = np.full(len(candidates), np.inf, dtype=float)
        row_error[finite] = np.abs(
            units[finite, row_axis] - rounded_rows[finite])
        col_min = float(np.min(grid[:, col_axis]))
        col_max = float(np.max(grid[:, col_axis]))
        aligned = (
            finite & (row_error < 0.18) &
            (rounded_rows >= row_values[0]) &
            (rounded_rows <= row_values[-1]) &
            (units[:, col_axis] >= col_min - 0.75) &
            (units[:, col_axis] <= col_max + 0.75))

        groups = {}
        for row in row_values:
            ids = np.where(aligned & (rounded_rows == row))[0]
            # final_points is already spatially merged, but retain an explicit
            # guard so repeated template responses cannot inflate row counts.
            selected = []
            for idx in ids[np.argsort(candidates[ids, 0])]:
                if (not selected or min(np.linalg.norm(
                        candidates[int(idx)] - candidates[old])
                        for old in selected) > 3.0):
                    selected.append(int(idx))
            groups[int(row)] = selected

        counts = np.asarray([len(groups[int(row)]) for row in row_values])
        if len(counts) == 0:
            continue
        target_cols = int(np.rint(np.median(counts)))
        if (target_cols <= current_cols or
                target_cols > current_cols + 3 or target_cols < 5):
            continue
        full_rows = [int(row) for row in row_values
                     if len(groups[int(row)]) == target_cols]
        if len(full_rows) < max(4, int(np.ceil(0.45 * len(row_values)))):
            continue

        # Determine the across-row ordering from existing one-cell neighbours.
        mapping = {tuple(g): p for g, p in zip(grid, img_pts)}
        step_vectors = []
        for key, p0 in mapping.items():
            neighbour = list(key)
            neighbour[col_axis] += 1
            p1 = mapping.get(tuple(neighbour))
            if p1 is None:
                continue
            vec = np.asarray(p1 - p0, dtype=float)
            if abs(vec[0]) >= abs(vec[1]):
                if vec[0] < 0:
                    vec *= -1.0
            elif vec[1] < 0:
                vec *= -1.0
            step_vectors.append(vec / max(np.linalg.norm(vec), 1e-9))
        if len(step_vectors) < 4:
            continue
        col_direction = np.median(np.asarray(step_vectors), axis=0)
        col_direction /= max(np.linalg.norm(col_direction), 1e-9)

        seed_img, seed_units = [], []
        for row in full_rows:
            ids = groups[row]
            order = sorted(ids, key=lambda idx: float(
                np.dot(candidates[idx], col_direction)))
            for physical_col, idx in enumerate(order):
                unit = np.zeros(2, dtype=float)
                unit[row_axis] = float(row)
                unit[col_axis] = float(physical_col)
                seed_img.append(candidates[idx])
                seed_units.append(unit)
        seed_img = np.asarray(seed_img, dtype=float)
        seed_world = np.asarray(seed_units, dtype=float) * grid_spacing
        new_H = compute_homography(seed_world, seed_img)
        if new_H is None:
            continue
        _, seed_reproj = compute_reprojection_error(
            new_H, seed_world, seed_img)
        if seed_reproj > 2.5:
            continue

        # Match every cell to an actual saddle candidate; unmatched cells stay
        # absent, preserving the requested gaps under occlusion or weak score.
        all_units = []
        for row in row_values:
            for physical_col in range(target_cols):
                unit = np.zeros(2, dtype=float)
                unit[row_axis] = float(row)
                unit[col_axis] = float(physical_col)
                all_units.append(unit)
        all_units = np.asarray(all_units, dtype=float)
        all_world = all_units * grid_spacing
        tree = KDTree(candidates)
        matched_img = seed_img
        matched_world = seed_world
        for _ in range(3):
            projected = _project_world_points(new_H, all_world)
            if projected is None or not np.isfinite(projected).all():
                break
            k = min(4, len(candidates))
            distances, indices = tree.query(projected, k=k)
            if k == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]
            proposals = []
            for grid_idx in range(len(projected)):
                for neighbour_idx in range(k):
                    distance = float(distances[grid_idx, neighbour_idx])
                    if distance <= 8.0:
                        proposals.append((distance, grid_idx,
                                          int(indices[grid_idx, neighbour_idx])))
            used_grid, used_candidate = set(), set()
            chosen_grid, chosen_candidate = [], []
            for _, grid_idx, candidate_idx in sorted(proposals):
                if grid_idx in used_grid or candidate_idx in used_candidate:
                    continue
                used_grid.add(grid_idx)
                used_candidate.add(candidate_idx)
                chosen_grid.append(grid_idx)
                chosen_candidate.append(candidate_idx)
            if len(chosen_grid) < len(seed_img):
                break
            trial_img = candidates[np.asarray(chosen_candidate, dtype=int)]
            trial_world = all_world[np.asarray(chosen_grid, dtype=int)]
            trial_H = compute_homography(trial_world, trial_img)
            if trial_H is None:
                break
            matched_img, matched_world, new_H = trial_img, trial_world, trial_H

        if len(matched_img) < len(img_pts) + max(5, len(row_values)):
            continue
        p90_line, _, bad_groups = chessboard_line_residual_summary(
            matched_img, matched_world)
        _, reproj = compute_reprojection_error(new_H, matched_world, matched_img)
        if bad_groups > 0 or p90_line > 3.0 or reproj > 2.5:
            continue

        new_qual = max(old_qual, calculate_chessboard_quality(
            matched_img, matched_world))
        rebuilt[board_idx] = (
            matched_img, matched_world, new_H, used_idx, new_qual)
        print(f"  [宽棋盘完整行重建] 棋盘#{board_idx + 1}: "
              f"{current_cols}列 -> {target_cols}列, "
              f"真实鞍点 {len(img_pts)} -> {len(matched_img)}")

    return rebuilt


def detect_two_column_strip_fallback(points, existing_boards, min_pairs=4):
    """Recover a partially occluded two-column strip from repeated row pairs.

    This is used when four-point homography seeding fails although the image
    still contains several genuine two-column saddle pairs.  Points already
    owned by an existing board are excluded, preventing a wider board from
    being rediscovered as a two-column sub-board.
    """
    if len(points) < 2 * min_pairs:
        return []
    available = _available_strip_points_mask(points, existing_boards)

    ids = np.where(available)[0]
    row_pairs = []
    for ai in range(len(ids)):
        i = int(ids[ai])
        for aj in range(ai + 1, len(ids)):
            j = int(ids[aj])
            p0, p1 = points[i], points[j]
            if p1[0] < p0[0]:
                i0, i1 = j, i
                p0, p1 = p1, p0
            else:
                i0, i1 = i, j
            dx = float(p1[0] - p0[0])
            dy = float(abs(p1[1] - p0[1]))
            if 18.0 <= dx <= 75.0 and dy <= max(5.0, 0.18 * dx):
                row_pairs.append({
                    'idx': (i0, i1),
                    'mid': 0.5 * (p0 + p1),
                    'width': dx,
                })
    if len(row_pairs) < min_pairs:
        return []

    row_pairs.sort(key=lambda r: r['mid'][1])
    best_chain = []
    for seed in range(len(row_pairs)):
        chain = [row_pairs[seed]]
        used_point_ids = set(row_pairs[seed]['idx'])
        cur = row_pairs[seed]
        while True:
            choices = []
            for candidate in row_pairs:
                if candidate['mid'][1] <= cur['mid'][1] + 15.0:
                    continue
                if used_point_ids.intersection(candidate['idx']):
                    continue
                delta = candidate['mid'] - cur['mid']
                dy = float(delta[1])
                if dy < 22.0 or dy > 75.0:
                    continue
                if abs(float(delta[0])) > 0.38 * dy + 3.0:
                    continue
                ratio = candidate['width'] / max(cur['width'], 1e-9)
                if ratio < 0.65 or ratio > 1.45:
                    continue
                cost = abs(delta[0]) + 0.25 * dy + 8.0 * abs(np.log(ratio))
                choices.append((cost, candidate))
            if not choices:
                break
            _, cur = min(choices, key=lambda item: item[0])
            chain.append(cur)
            used_point_ids.update(cur['idx'])
        if len(chain) > len(best_chain):
            best_chain = chain

    if len(best_chain) < min_pairs:
        return []
    # Reject chains with implausibly irregular row spacing.
    row_y = np.asarray([r['mid'][1] for r in best_chain])
    steps = np.diff(row_y)
    if len(steps) and np.max(steps) > 1.65 * max(np.median(steps), 1.0):
        return []

    # Start with complete row pairs, then continue through rows where only one
    # column remains visible.  Only an actually detected point is inserted;
    # the hidden counterpart is left absent.
    rows_data = [[(pair['idx'][0], 0), (pair['idx'][1], 1)] for pair in best_chain]
    used_idx = set(idx for row in rows_data for idx, _ in row)
    if len(best_chain) >= 3 and len(steps):
        step_y = float(np.median(steps))
        mid_x = np.asarray([r['mid'][0] for r in best_chain])
        drift_x = float(np.median(np.diff(mid_x))) if len(mid_x) > 1 else 0.0
        width = float(np.median([r['width'] for r in best_chain]))
        last_mid = best_chain[-1]['mid'].copy()
        for _ in range(12):
            predicted_mid = last_mid + np.array([drift_x, step_y])
            predicted = [predicted_mid + np.array([-0.5 * width, 0.0]),
                         predicted_mid + np.array([0.5 * width, 0.0])]
            row_hits = []
            for col, target in enumerate(predicted):
                candidates = []
                for idx in ids:
                    idx = int(idx)
                    if idx in used_idx:
                        continue
                    error = float(np.linalg.norm(points[idx] - target))
                    if error <= 0.34 * step_y + 3.0:
                        candidates.append((error, idx))
                if candidates:
                    _, idx = min(candidates)
                    row_hits.append((idx, col))
            if not row_hits:
                break
            # Do not let the same point fill both columns.
            unique_hits = []
            seen = set()
            for hit in sorted(row_hits, key=lambda item: item[1]):
                if hit[0] not in seen:
                    unique_hits.append(hit)
                    seen.add(hit[0])
            rows_data.append(unique_hits)
            used_idx.update(idx for idx, _ in unique_hits)
            if len(unique_hits) == 2:
                last_mid = 0.5 * (points[unique_hits[0][0]] + points[unique_hits[1][0]])
            elif unique_hits[0][1] == 0:
                last_mid = points[unique_hits[0][0]] + np.array([0.5 * width, 0.0])
            else:
                last_mid = points[unique_hits[0][0]] - np.array([0.5 * width, 0.0])

    if len(rows_data) < max(5, min_pairs):
        return []
    img_pts = []
    world_pts = []
    for row, entries in enumerate(rows_data):
        for idx, col in entries:
            img_pts.append(points[idx])
            world_pts.append([col * 15.0, row * 15.0])
    img_pts = np.asarray(img_pts, dtype=float)
    world_pts = np.asarray(world_pts, dtype=float)
    H = compute_homography(world_pts, img_pts)
    if H is None:
        return []
    print(f"  [两列回退] 从 {len(best_chain)} 个完整行对扩展到 {len(rows_data)} 行，"
          f"共 {len(img_pts)} 个真实鞍点")
    return [(img_pts, world_pts, H, used_idx)]


def detect_one_sided_strip_fallback(points, existing_boards, min_chain=5,
                                    grid_spacing=15.0, point_scores=None):
    """Recover a two-column strip when all but one row lose one column.

    This fallback requires both a long, regularly spaced column and at least
    one genuine same-row partner proving the strip width.  Only detected
    saddles are returned; the hidden column is used solely as an affine
    geometric prior and is never added as a display node.
    """
    if len(points) < min_chain + 1:
        return []
    available = _available_strip_points_mask(points, existing_boards)
    ids = np.where(available)[0]
    if len(ids) < min_chain + 1:
        return []

    best = None
    for ai in range(len(ids)):
        i = int(ids[ai])
        for aj in range(len(ids)):
            j = int(ids[aj])
            if points[j, 1] <= points[i, 1]:
                continue
            step = points[j] - points[i]
            step_len = float(np.linalg.norm(step))
            if (step_len < 28.0 or step_len > 75.0 or
                    abs(float(step[0])) > 0.35 * abs(float(step[1])) + 2.0):
                continue

            tol = max(3.5, 0.13 * step_len)
            hits = {}
            for k in range(-12, 13):
                target = points[i] + k * step
                delta = points[ids] - target
                dist = np.linalg.norm(delta, axis=1)
                q = int(np.argmin(dist))
                if dist[q] <= tol:
                    hits[int(ids[q])] = k
            if len(hits) < min_chain:
                continue

            chain_ids = sorted(hits, key=lambda idx: points[idx, 1])
            chain = points[chain_ids]
            row_no = np.arange(len(chain), dtype=float)
            fit_x = np.polyfit(row_no, chain[:, 0], 1)
            fit_y = np.polyfit(row_no, chain[:, 1], 1)
            fitted = np.column_stack([np.polyval(fit_x, row_no),
                                      np.polyval(fit_y, row_no)])
            residual = np.linalg.norm(chain - fitted, axis=1)
            row_vec = np.array([fit_x[0], fit_y[0]], dtype=float)
            row_step = float(np.linalg.norm(row_vec))
            if (row_step < 28.0 or row_step > 75.0 or
                    np.percentile(residual, 90) > 3.5):
                continue
            gaps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            if (len(gaps) and
                    (np.min(gaps) < 0.72 * row_step or
                     np.max(gaps) > 1.30 * row_step)):
                continue

            partners = []
            chain_set = set(chain_ids)
            for row, idx in enumerate(chain_ids):
                p = points[idx]
                choices = []
                for other in ids:
                    other = int(other)
                    if other in chain_set:
                        continue
                    if (point_scores is not None and
                            float(point_scores[other]) < 0.012):
                        continue
                    dx, dy = points[other] - p
                    width = abs(float(dx))
                    if (18.0 <= width <= 75.0 and
                            abs(float(dy)) <= max(5.0, 0.18 * width)):
                        choices.append((abs(float(dy)), width, other,
                                        1 if dx > 0 else -1, row))
                if choices:
                    partners.append(min(choices))
            if not partners:
                continue

            side = 1 if sum(p[3] > 0 for p in partners) >= \
                        sum(p[3] < 0 for p in partners) else -1
            same_side = [p for p in partners if p[3] == side]
            width_med = float(np.median([p[1] for p in same_side]))
            consistent = [p for p in same_side
                          if 0.68 * width_med <= p[1] <= 1.38 * width_med]
            if not consistent:
                continue
            score = 3.0 * len(chain_ids) + 2.0 * len(consistent) - \
                    float(np.mean(residual))
            if best is None or score > best[0]:
                best = (score, chain_ids, consistent, side, row_vec)

    if best is None:
        return []
    _, chain_ids, partners, side, row_vec = best
    partner_by_row = {}
    for item in partners:
        _, _, idx, _, row = item
        partner_by_row.setdefault(int(row), int(idx))

    img_pts, world_pts, used_idx = [], [], set()
    chain_col = 0 if side > 0 else 1
    other_col = 1 - chain_col
    for row, idx in enumerate(chain_ids):
        img_pts.append(points[idx])
        world_pts.append([chain_col * grid_spacing, row * grid_spacing])
        used_idx.add(int(idx))
        if row in partner_by_row:
            other = partner_by_row[row]
            img_pts.append(points[other])
            world_pts.append([other_col * grid_spacing, row * grid_spacing])
            used_idx.add(int(other))
    img_pts = np.asarray(img_pts, dtype=float)
    world_pts = np.asarray(world_pts, dtype=float)

    first_chain = points[chain_ids[0]]
    sample_row = min(partner_by_row)
    sample_partner = points[partner_by_row[sample_row]]
    sample_chain = points[chain_ids[sample_row]]
    col_vec = (sample_partner - sample_chain) * side
    origin = first_chain - chain_col * col_vec
    H = np.array([[col_vec[0] / grid_spacing, row_vec[0] / grid_spacing, origin[0]],
                  [col_vec[1] / grid_spacing, row_vec[1] / grid_spacing, origin[1]],
                  [0.0, 0.0, 1.0]], dtype=float)
    print(f"  [单侧两列回退] {len(chain_ids)} 行稳定单列 + "
          f"{len(partner_by_row)} 个真实同行伙伴，共 {len(img_pts)} 个真实鞍点")
    return [(img_pts, world_pts, H, used_idx)]


def detect_x_guided_single_column_fallback(points, existing_boards, img_gray,
                                            point_scores=None,
                                            grid_spacing=15.0):
    """Recover visible saddles when an occluded strip leaves no full row pair.

    A complete butterfly above the run supplies the missing evidence that the
    remaining collinear responses belong to a physical two-column target.  No
    hidden saddle is generated: the returned display topology contains only
    actually detected points in the visible column, with row gaps preserved.
    """
    if len(points) < 3:
        return []
    available = _available_strip_points_mask(points, existing_boards)
    ids = np.where(available)[0]
    if len(ids) < 3:
        return []
    pts = np.asarray(points, dtype=float)
    _, image_width = img_gray.shape
    best = None

    for seed_idx in ids:
        seed_idx = int(seed_idx)
        seed = pts[seed_idx]
        # X-guided strip recovery is only valid on the central two-column
        # targets.  The outer edge of a broad checkerboard can otherwise act
        # as a high-contrast seed and be duplicated as a one-column board
        # (hewei196).
        if seed[0] < 0.22 * image_width or seed[0] > 0.92 * image_width:
            continue
        if _is_broad_board_lattice_extension(seed, existing_boards):
            continue
        appearance = max(_butterfly_contrast_score(
            img_gray, seed[0], seed[1], r) for r in (14.0, 18.0, 22.0, 26.0))
        if appearance < 0.70:
            continue
        delta = pts[ids] - seed
        below_local = np.where((delta[:, 1] >= 70.0) & (delta[:, 1] <= 430.0) &
                               (np.abs(delta[:, 0]) <= 50.0))[0]
        below_ids = ids[below_local]
        if len(below_ids) < 2:
            continue

        for ai in range(len(below_ids)):
            for aj in range(ai + 1, len(below_ids)):
                i = int(below_ids[ai])
                j = int(below_ids[aj])
                if pts[j, 1] < pts[i, 1]:
                    i, j = j, i
                pair_gap = float(pts[j, 1] - pts[i, 1])
                if pair_gap < 35.0 or pair_gap > 130.0:
                    continue
                if abs(float(pts[j, 0] - pts[i, 0])) > 20.0:
                    continue
                first_gap = float(pts[i, 1] - seed[1])
                step_prior = float(np.clip(first_gap / 3.6, 28.0, 75.0))
                row_jump = int(np.clip(round(pair_gap / step_prior), 1, 4))
                row_step = pair_gap / row_jump
                if row_step < 28.0 or row_step > 75.0:
                    continue
                lateral = abs(float(0.5 * (pts[i, 0] + pts[j, 0]) - seed[0]))
                if lateral > 25.0 or first_gap < 2.0 * row_step:
                    continue

                drift = float(pts[j, 0] - pts[i, 0]) / row_jump
                rows = {}
                for idx in ids:
                    idx = int(idx)
                    if pts[idx, 1] < pts[i, 1] - 0.25 * row_step:
                        continue
                    row = int(round((pts[idx, 1] - pts[i, 1]) / row_step))
                    if row < 0 or row > 12:
                        continue
                    pred = pts[i] + np.array([row * drift, row * row_step])
                    error = float(np.linalg.norm(pts[idx] - pred))
                    if error > max(5.0, 0.16 * row_step):
                        continue
                    candidate_score = (float(point_scores[idx])
                                       if point_scores is not None else 1.0)
                    rank = candidate_score - 0.02 * error
                    if row not in rows or rank > rows[row][0]:
                        rows[row] = (rank, idx)
                if len(rows) < 2 or 0 not in rows or row_jump not in rows:
                    continue
                chain_ids = [rows[row][1] for row in sorted(rows)]
                score = 4.0 * len(chain_ids) + appearance - 0.02 * lateral
                if best is None or score > best[0]:
                    best = (score, chain_ids, sorted(rows), seed_idx, appearance)

    if best is None:
        return []
    _, chain_ids, row_ids, seed_idx, appearance = best
    img_pts = pts[chain_ids]
    # The butterfly seed alone is not sufficient to prove a separate narrow
    # target.  On a severely tilted broad board an exposed outer checker
    # corner can have a high butterfly score while the following candidates
    # are simply an omitted boundary column (hewei207).  Reject the fallback
    # when the recovered run itself coherently lies on an existing broad-board
    # lattice.  Testing the run, rather than only the seed, is robust to the
    # larger extrapolation error just outside the board hull.
    broad_lattice_hits = sum(
        _is_broad_board_lattice_extension(point, existing_boards)
        for point in img_pts
    )
    if broad_lattice_hits >= max(3, int(np.ceil(0.80 * len(img_pts)))):
        print(f"  [X引导单列回退] 跳过宽棋盘边界列 "
              f"({broad_lattice_hits}/{len(img_pts)}点落在已有宽棋盘网格)")
        return []
    world_pts = np.column_stack([
        np.zeros(len(row_ids), dtype=float),
        np.asarray(row_ids, dtype=float) * grid_spacing
    ])
    used_idx = set(int(idx) for idx in chain_ids)
    print(f"  [X引导单列回退] 恢复 {len(img_pts)} 个真实鞍点，"
          f"保留行号 {row_ids}，蝴蝶分={appearance:.3f}")
    return [(img_pts, world_pts, None, used_idx)]


def _point_line_residuals(pts):
    if len(pts) < 3:
        return np.zeros(len(pts))
    centered = pts - np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    return np.abs(centered @ normal)


def grid_line_residuals(img_pts, world_pts):
    residual_sum = np.zeros(len(img_pts), dtype=float)
    residual_count = np.zeros(len(img_pts), dtype=float)
    rounded = np.round(world_pts / 15.0).astype(int)
    for axis in (0, 1):
        for value in np.unique(rounded[:, axis]):
            idx = np.where(rounded[:, axis] == value)[0]
            if len(idx) < 4:
                continue
            residual_sum[idx] += _point_line_residuals(img_pts[idx])
            residual_count[idx] += 1
    residuals = np.zeros(len(img_pts), dtype=float)
    valid = residual_count > 0
    residuals[valid] = residual_sum[valid] / residual_count[valid]
    return residuals, valid


def chessboard_line_residual_summary(img_pts, world_pts):
    img_pts, world_pts = _unique_world_points(img_pts, world_pts)
    if len(img_pts) < 4:
        return 1e9, 1e9, 99
    rounded = np.round(world_pts / 15.0).astype(int)
    medians = []
    maxes = []
    for axis in (0, 1):
        for value in np.unique(rounded[:, axis]):
            idx = np.where(rounded[:, axis] == value)[0]
            if len(idx) < 4:
                continue
            residuals = _point_line_residuals(img_pts[idx])
            medians.append(float(np.median(residuals)))
            maxes.append(float(np.max(residuals)))
    if not medians:
        return 1e9, 1e9, 99
    medians = np.asarray(medians, dtype=float)
    maxes = np.asarray(maxes, dtype=float)
    return float(np.percentile(medians, 90)), float(np.max(maxes)), int(np.sum(medians > 6.0))


def filter_chessboard_line_outliers(img_pts, world_pts, threshold=7.5, min_points=9):
    img_pts, world_pts = _unique_world_points(img_pts, world_pts)
    residuals, valid = grid_line_residuals(img_pts, world_pts)
    keep = ~valid | (residuals <= threshold)
    if np.sum(keep) < min_points:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)
    removed = np.sum(~keep)
    if removed > 0:
        print(f"  最终行列离群点剔除 {removed} 个")
    return img_pts[keep], world_pts[keep], compute_homography(world_pts[keep], img_pts[keep])


def _project_world_points(H, world_pts):
    if H is None or len(world_pts) == 0:
        return None
    xy = np.column_stack([world_pts[:, 0], world_pts[:, 1], np.ones(len(world_pts))])
    uv_h = (H @ xy.T).T
    valid = np.abs(uv_h[:, 2]) > 1e-9
    projected = np.full((len(world_pts), 2), np.nan, dtype=float)
    projected[valid] = uv_h[valid, :2] / uv_h[valid, 2, np.newaxis]
    return projected


def _homography_has_grid_scale(H, world_pts, grid_spacing=15.0):
    """Reject a recomputed homography whose row/column step has collapsed."""
    if H is None or len(world_pts) == 0:
        return False
    base = np.min(np.asarray(world_pts, dtype=float), axis=0)
    probes = np.array([base,
                       base + [grid_spacing, 0.0],
                       base + [0.0, grid_spacing]], dtype=float)
    projected = _project_world_points(H, probes)
    if projected is None or not np.isfinite(projected).all():
        return False
    steps = np.linalg.norm(projected[1:] - projected[0], axis=1)
    return bool(np.all((steps >= 5.0) & (steps <= 300.0)))


def _projected_grid_spacing(projected, x_units, y_units):
    if projected is None or len(projected) == 0:
        return 20.0
    grid = projected.reshape(len(y_units), len(x_units), 2)
    distances = []
    if len(x_units) > 1:
        diff_x = np.linalg.norm(np.diff(grid, axis=1), axis=2)
        distances.extend(diff_x[np.isfinite(diff_x)].ravel())
    if len(y_units) > 1:
        diff_y = np.linalg.norm(np.diff(grid, axis=0), axis=2)
        distances.extend(diff_y[np.isfinite(diff_y)].ravel())
    if not distances:
        return 20.0
    return float(np.median(distances))


def _trim_sparse_border_grid(img_pts, world_pts, min_fill=0.25, min_points=9):  # 降低边界填充率要求
    img_pts, world_pts = _unique_world_points(img_pts, world_pts)
    if len(img_pts) < min_points:
        return img_pts, world_pts
    original_img = img_pts
    original_world = world_pts
    changed = True
    while changed and len(img_pts) >= min_points:
        changed = False
        snapped = np.round(world_pts / 15.0).astype(int)
        cols = np.unique(snapped[:, 0])
        rows = np.unique(snapped[:, 1])
        if len(cols) < 2 or len(rows) < 2:
            break
        keep = np.ones(len(img_pts), dtype=bool)
        # 边界行/列的最小点数不能超过该方向的总列/行数，否则对细长棋盘格会过度修剪
        min_row_count = min(len(cols), max(1, int(np.ceil(len(cols) * min_fill))))
        min_col_count = min(len(rows), max(1, int(np.ceil(len(rows) * min_fill))))
        for row in (rows[0], rows[-1]):
            idx = np.where(snapped[:, 1] == row)[0]
            if len(idx) < min_row_count:
                keep[idx] = False
        for col in (cols[0], cols[-1]):
            idx = np.where(snapped[:, 0] == col)[0]
            if len(idx) < min_col_count:
                keep[idx] = False
        if np.sum(keep) < min_points:
            return original_img, original_world
        if np.sum(keep) < len(img_pts):
            img_pts = img_pts[keep]
            world_pts = world_pts[keep]
            changed = True
    return img_pts, world_pts



def interpolate_missing_grid_points(img_pts, world_pts, min_neighbors=3):
    """根据已检测的棋盘格网格，对内部缺失的网格点进行单应插值填补。
    仅当缺失点至少有 min_neighbors 个已检测的 4-邻接邻居时才添加，
    以减少在反光/低对比区域引入错误点的风险。
    """
    img_pts, world_pts = _unique_world_points(img_pts, world_pts)
    if len(img_pts) < 6:
        return img_pts, world_pts
    snapped = np.round(world_pts / 15.0).astype(int)
    existing = set(map(tuple, snapped))
    cols = np.unique(snapped[:, 0])
    rows = np.unique(snapped[:, 1])
    if len(cols) < 2 or len(rows) < 2:
        return img_pts, world_pts

    col_min, col_max = int(cols.min()), int(cols.max())
    row_min, row_max = int(rows.min()), int(rows.max())
    full_cols = np.arange(col_min, col_max + 1)
    full_rows = np.arange(row_min, row_max + 1)
    xx, yy = np.meshgrid(full_cols, full_rows)
    full_world = np.column_stack([xx.ravel() * 15.0, yy.ravel() * 15.0])

    H = compute_homography(world_pts, img_pts)
    if H is None:
        return img_pts, world_pts
    projected = _project_world_points(H, full_world)
    if projected is None:
        return img_pts, world_pts

    new_img, new_world = [], []
    for i, (c, r) in enumerate(zip(xx.ravel(), yy.ravel())):
        if (c, r) in existing:
            continue
        # 统计 4-邻接已检测邻居数
        neighbors = sum(
            1 for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if (c + dc, r + dr) in existing
        )
        # 边界点（最外圈）放宽要求：只需 1 个邻居即可推断，
        # 避免细长棋盘格的顶角/边角因遮挡或漏检而无法补齐。
        is_boundary = (c == col_min or c == col_max or
                       r == row_min or r == row_max)
        required = 1 if is_boundary else min_neighbors
        if neighbors >= required:
            pt = projected[i]
            if np.isfinite(pt).all():
                new_img.append(pt)
                new_world.append(full_world[i])


    if not new_img:
        return img_pts, world_pts
    img_pts = np.vstack([img_pts, np.asarray(new_img, dtype=float)])
    world_pts = np.vstack([world_pts, np.asarray(new_world, dtype=float)])
    return _unique_world_points(img_pts, world_pts)


def complete_chessboard_from_candidates(img_pts, world_pts, candidate_pts, min_points=9, expand_margin=1):
    img_pts, world_pts = _unique_world_points(img_pts, world_pts)
    candidate_pts = np.asarray(candidate_pts, dtype=float)
    if len(img_pts) < 4 or len(candidate_pts) < min_points:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    snapped = np.round(world_pts / 15.0).astype(int)
    margin = max(0, int(expand_margin))
    x_units = np.arange(np.min(snapped[:, 0]) - margin, np.max(snapped[:, 0]) + margin + 1)
    y_units = np.arange(np.min(snapped[:, 1]) - margin, np.max(snapped[:, 1]) + margin + 1)
    if len(x_units) < 2 or len(y_units) < 2:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    xx, yy = np.meshgrid(x_units, y_units)
    grid_world = np.column_stack([xx.ravel() * 15.0, yy.ravel() * 15.0])
    tree = KDTree(candidate_pts)

    best_img = img_pts
    best_world = world_pts
    H = compute_homography(world_pts, img_pts)
    if H is None:
        return img_pts, world_pts, H

    for _ in range(3):
        projected = _project_world_points(H, grid_world)
        if projected is None:
            break
        finite = np.isfinite(projected).all(axis=1)
        if np.sum(finite) < min_points:
            break

        spacing = _projected_grid_spacing(projected, x_units, y_units)
        match_threshold = max(6.0, min(18.0, spacing * 0.42))
        k = min(4, len(candidate_pts))
        dists, idxs = tree.query(projected[finite], k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        finite_grid_idx = np.where(finite)[0]
        proposals = []
        for local_i, grid_i in enumerate(finite_grid_idx):
            for neighbor_i in range(k):
                dist = float(dists[local_i, neighbor_i])
                if dist <= match_threshold:
                    proposals.append((dist, int(grid_i), int(idxs[local_i, neighbor_i])))

        proposals.sort(key=lambda item: item[0])
        used_grid = set()
        used_candidate = set()
        matched_grid = []
        matched_candidate = []
        for _, grid_i, cand_i in proposals:
            if grid_i in used_grid or cand_i in used_candidate:
                continue
            used_grid.add(grid_i)
            used_candidate.add(cand_i)
            matched_grid.append(grid_i)
            matched_candidate.append(cand_i)

        if len(matched_grid) < max(min_points, int(len(img_pts) * 0.50)):  # 放宽：允许H不精确时仍继续迭代
            break

        new_world = grid_world[np.array(matched_grid)]
        new_img = candidate_pts[np.array(matched_candidate)]
        new_H = compute_homography(new_world, new_img)
        if new_H is None:
            break
        best_img, best_world, H = new_img, new_world, new_H

    order = np.lexsort([best_world[:, 0], best_world[:, 1]])
    best_img = best_img[order]
    best_world = best_world[order]
    best_img, best_world = _trim_sparse_border_grid(best_img, best_world, min_points=min_points)
    H = compute_homography(best_world, best_img)
    if len(best_img) != len(img_pts):
        print(f"  单应网格重建 {len(img_pts)} -> {len(best_img)} 个交叉点")
    return best_img, best_world, H


def adjust_skipped_grid_axis(img_pts, world_pts, candidate_pts, min_inside=4):
    img_pts, world_pts = _unique_world_points(img_pts, world_pts)
    candidate_pts = np.asarray(candidate_pts, dtype=float)
    if len(img_pts) < 9 or len(candidate_pts) <= len(img_pts):
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    polygon, _ = get_chessboard_polygon(img_pts)
    if polygon is None:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    inside = polygon.contains_points(candidate_pts)
    tree = KDTree(img_pts)
    nearest_dist, _ = tree.query(candidate_pts, k=1)
    unused = candidate_pts[inside & (nearest_dist > 3.0)]
    if len(unused) < min_inside:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    units = world_pts / 15.0
    units = units - np.min(units, axis=0)
    best = None
    # A seed can grow with one or two genuine saddles between consecutive seed
    # nodes.  The latter case needs x3, not x2 (hewei180).  Evaluate both axes
    # independently and together.  Requiring support in every intermediate
    # residue class below prevents an x2 lattice from being over-expanded to
    # x3 merely because a few clutter candidates happen to project cleanly.
    scale_patterns = [
        (sx, sy)
        for sx in (1.0, 2.0, 3.0)
        for sy in (1.0, 2.0, 3.0)
        if not (sx == 1.0 and sy == 1.0)
    ]
    for scales in scale_patterns:
        trial_units = units * np.asarray(scales, dtype=float)
        H_img_to_grid = compute_homography(img_pts, trial_units)
        if H_img_to_grid is None:
            continue
        projected = _project_world_points(H_img_to_grid, unused)
        if projected is None:
            continue
        max_units = np.max(trial_units, axis=0)
        close_count = 0
        scaled_axes = [axis for axis, scale in enumerate(scales) if scale > 1.0]
        residue_hits = {
            axis: {residue: 0 for residue in range(1, int(scales[axis]))}
            for axis in scaled_axes
        }
        errors = []
        for point in projected:
            if not np.isfinite(point).all():
                continue
            rounded = np.round(point).astype(int)
            if np.any(rounded < -1) or rounded[0] > max_units[0] + 1 or rounded[1] > max_units[1] + 1:
                continue
            err = float(np.linalg.norm(point - rounded))
            if err < 0.28:
                new_on_scaled_axis = False
                for axis in scaled_axes:
                    residue = int(rounded[axis] % int(scales[axis]))
                    if residue != 0:
                        residue_hits[axis][residue] += 1
                        new_on_scaled_axis = True
                if not new_on_scaled_axis:
                    continue
                close_count += 1
                errors.append(err)
        required = max(min_inside, int(len(unused) * 0.18))
        axes_supported = True
        for axis in scaled_axes:
            per_residue_required = max(
                2, int(np.ceil(required / max(1, int(scales[axis]) - 1))))
            if any(count < per_residue_required
                   for count in residue_hits[axis].values()):
                axes_supported = False
                break
        if axes_supported and close_count >= max(min_inside, int(len(unused) * 0.35)):
            # Prefer the simpler/smaller expansion when support is equal.
            complexity = (1.0 + 0.08 * (len(scaled_axes) - 1) +
                          0.05 * sum(scale - 2.0 for scale in scales if scale > 1.0))
            score = close_count / ((np.median(errors) + 1e-6) * complexity)
            if best is None or score > best[0]:
                best = (score, scales, trial_units)

    if best is None:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    _, scales, adjusted_units = best
    adjusted_units = adjusted_units - np.min(adjusted_units, axis=0)
    adjusted_world = adjusted_units * 15.0
    axes_text = []
    if scales[0] > 1.0:
        axes_text.append('X/列')
    if scales[1] > 1.0:
        axes_text.append('Y/行')
    print(f"  跳格修正: {'+'.join(axes_text)} 轴加密")
    return img_pts, adjusted_world, compute_homography(adjusted_world, img_pts)


def get_mark_cord(xc, corner, xy1, uv1, HH0, nn, spij3, spij7, jilu, xcsp, Im0, mm, img_gray):
    """X 角点（黑白半圆标记）检测包装函数，如果失败则返回原始棋盘格结果"""
    try:
        return get_mark_cord_m(xc, corner, xy1, uv1, HH0, nn, spij3, spij7, jilu, xcsp, Im0, mm, img_gray)
    except Exception as exc:
        print(f"  警告: 标记点识别失败，保留棋盘格结果: {exc}")
        return xy1, uv1, jilu


# ===================== X角点检测：适用于归零世界坐标 =====================
def _detect_x_corners_on_board(img_pts, world_pts, H, all_saddle_pts, corner,
                                h, Im0, spij3, grid_spacing=15.0,
                                full_col_min=None, full_col_max=None,
                                full_row_min=None, full_row_max=None):
    """
    在单个棋盘格上检测X形半圆标记（最多4个，分布在四个角）。
    X标记位于棋盘格网格外侧半格处（即棋盘格白色底板的四个实际外角）。

    :param full_col_min/max, full_row_min/max:
        完整棋盘格网格范围（用于修正遮挡导致的范围收缩）。
        若未提供，则回退到 detected 范围。
        角点类型：1=左上, 2=右上, 3=左下, 4=右下
    """
    max_index = len(Im0)
    if H is None or len(world_pts) < 4 or len(all_saddle_pts) == 0:
        return []

    # 1. 确定网格范围
    snapped = np.round(world_pts / grid_spacing).astype(int)
    col_min, col_max = int(np.min(snapped[:, 0])), int(np.max(snapped[:, 0]))
    row_min, row_max = int(np.min(snapped[:, 1])), int(np.max(snapped[:, 1]))

    # 如果提供了完整范围，优先使用（修正遮挡导致的范围收缩）
    if full_col_min is not None:
        col_min = min(col_min, full_col_min) if full_col_min > col_min else full_col_min
    if full_col_max is not None:
        col_max = max(col_max, full_col_max) if full_col_max < col_max else full_col_max
    if full_row_min is not None:
        row_min = min(row_min, full_row_min) if full_row_min > row_min else full_row_min
    if full_row_max is not None:
        row_max = max(row_max, full_row_max) if full_row_max < row_max else full_row_max

    if col_max - col_min < 1 or row_max - row_min < 1:
        return []

    # 2. X标记世界坐标：位于棋盘格网格外侧半格处（底板实际外角）
    corner_world = np.array([
        [(col_min - 0.5) * grid_spacing, (row_min - 0.5) * grid_spacing],  # 1: 左上
        [(col_max + 0.5) * grid_spacing, (row_min - 0.5) * grid_spacing],  # 2: 右上
        [(col_min - 0.5) * grid_spacing, (row_max + 0.5) * grid_spacing],  # 3: 左下
        [(col_max + 0.5) * grid_spacing, (row_max + 0.5) * grid_spacing],  # 4: 右下
    ])
    corner_types = [1, 2, 3, 4]

    # 3. 投影到图像
    projected = _project_world_points(H, corner_world)
    if projected is None or len(projected) != 4:
        return []

    # 4. 获取棋盘格在图像中的方向（用于S值对角方向计算）
    # 从 H 投影两个相邻世界点来确定图像中棋盘格的轴方向
    origin_uv = _project_world_points(H, np.array([[0, 0]]))
    x_dir_uv = _project_world_points(H, np.array([[grid_spacing, 0]]))
    if origin_uv is None or x_dir_uv is None or len(origin_uv) == 0 or len(x_dir_uv) == 0:
        return []
    img_x_dir = (x_dir_uv[0] - origin_uv[0])
    img_x_norm = float(np.linalg.norm(img_x_dir))
    if img_x_norm < 1e-6:
        return []
    # 棋盘格X轴在图像中的单位方向
    uv0_global = img_x_dir / img_x_norm

    # 5. S值辅助函数（在任意像素位置计算S值对比度）
    def _compute_s_contrast_at(cx, cy):
        """在图像位置 (cx, cy) 计算X形标记的S值对比度。
        使用棋盘格的轴方向来确定4个对角方向。"""
        base_px = int(round(cx)) * h + int(round(cy))
        if base_px < 0 or base_px >= max_index:
            return 0.0
        ux, uy = uv0_global[0], uv0_global[1]
        # 4个对角方向（X形标记的4个扇区）
        n1 = base_px - int(round(7 * uy)) * h + int(round(7 * ux))
        n2 = base_px + int(round(7 * ux)) * h + int(round(7 * uy))
        n3 = base_px + int(round(7 * (ux + uy))) * h + int(round(7 * (-ux + uy)))
        n4 = base_px + int(round(7 * (ux - uy))) * h + int(round(7 * (ux + uy)))
        s_vals = []
        for ni in [n1, n2, n3, n4]:
            indices = ni + spij3
            mask = (indices >= 0) & (indices < max_index)
            s_vals.append(float(np.sum(Im0[indices[mask]])) if np.any(mask) else 0.0)
        smin, smax = min(s_vals), max(s_vals)
        if smax < 1e-6:
            return 0.0
        return smax / (smin + 1e-9)  # 对比度

    # 6. 对每个角：在投影点周围的小窗口内搜索最佳S值位置
    jilu_found = []
    spacing_img_val = np.mean(np.linalg.norm(
        np.diff(img_pts[:min(4, len(img_pts))], axis=0), axis=1)) if len(img_pts) >= 2 else 20.0
    # 搜索窗口半径：至少 10px，约半格间距（X标记可能偏离投影点达半格）
    search_offset = max(10, int(spacing_img_val * 0.55))
    img_w = max_index // h

    for i, (proj_pt, ctype) in enumerate(zip(projected, corner_types)):
        if not np.isfinite(proj_pt).all():
            continue
        px, py = float(proj_pt[0]), float(proj_pt[1])

        # 阶段1：粗搜索（步长2，快速定位）
        best_contrast = 1.3  # 较低阈值，保留更多候选
        best_pos = None
        for dx in range(-search_offset, search_offset + 1, 2):
            for dy in range(-search_offset, search_offset + 1, 2):
                cx, cy = px + dx, py + dy
                if cx < 0 or cy < 0 or cx >= img_w or cy >= h:
                    continue
                contrast = _compute_s_contrast_at(cx, cy)
                if contrast > best_contrast:
                    best_contrast = contrast
                    best_pos = (cx, cy)

        # 阶段2：在粗搜索最佳位置附近精搜索（步长1，精度更高）
        if best_pos is not None:
            cx0, cy0 = best_pos
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cx, cy = cx0 + dx, cy0 + dy
                    if cx < 0 or cy < 0 or cx >= img_w or cy >= h:
                        continue
                    contrast = _compute_s_contrast_at(cx, cy)
                    if contrast > best_contrast:
                        best_contrast = contrast
                        best_pos = (cx, cy)

        # 接受：对比度足够高且位置在图像范围内
        if best_pos is not None and best_contrast >= 1.5:
            jilu_found.append([ctype, best_pos[0], best_pos[1]])

    return jilu_found


# ===================== X角点遮挡推断函数（来自429(2)机制） =====================
def _infer_missing_corners_from_partial(jilu_partial, existing_world_pts,
                                         existing_img_pts, grid_spacing=15.0):
    """
    角点补全：基于已检测到的角点（1-3个，被遮挡时不足4个）和已有网格的几何关系，
    推算出缺失角点的图像坐标。仅用于内部推算，不用于可视化。

    核心思路：
    1. 从已有网格的bbox确定棋盘格世界坐标范围
    2. 4角点的世界坐标定义：X标记位于棋盘格网格外侧半格处（底板实际外角）
    3. 用H_grid将4角点世界坐标投影到图像
    4. 用已知角点对proj位置做仿射修正
    5. 返回补全后的4角点jilu

    :param jilu_partial: 部分角点记录 [mark_type, img_x, img_y]
    :param existing_world_pts: 现有棋盘格世界坐标
    :param existing_img_pts: 现有棋盘格图像坐标
    :param grid_spacing: 网格间距
    :return: 补全后的 jilu（4角点）或 None（无法补全）
    """
    if len(existing_img_pts) < 4 or len(existing_world_pts) < 4:
        return None

    # 1. 从已有世界坐标确定棋盘格范围
    world_grid = np.round(existing_world_pts / grid_spacing).astype(int)
    col_min, col_max = int(world_grid[:, 0].min()), int(world_grid[:, 0].max())
    row_min, row_max = int(world_grid[:, 1].min()), int(world_grid[:, 1].max())
    if col_max - col_min < 1 or row_max - row_min < 1:
        print(f"  [角点补全] 网格太小，无法定义角点")
        return None

    # 2. X角点世界坐标：位于棋盘格网格外侧半格处（底板实际四角）
    corner_world = {
        1: np.array([(col_min - 0.5) * grid_spacing, (row_min - 0.5) * grid_spacing],
                    dtype=np.float64),
        2: np.array([(col_max + 0.5) * grid_spacing, (row_min - 0.5) * grid_spacing],
                    dtype=np.float64),
        3: np.array([(col_min - 0.5) * grid_spacing, (row_max + 0.5) * grid_spacing],
                    dtype=np.float64),
        4: np.array([(col_max + 0.5) * grid_spacing, (row_max + 0.5) * grid_spacing],
                    dtype=np.float64),
    }

    # 3. 用已有网格点拟合 H_grid
    H_grid = compute_homography(existing_world_pts, existing_img_pts)
    if H_grid is None or not np.all(np.isfinite(H_grid)):
        print(f"  [角点补全] H_grid 拟合失败")
        return None

    # 4. 提取已知角点
    known = {}
    for rec in jilu_partial:
        mt = int(rec[0])
        if mt in {1, 2, 3, 4}:
            known[mt] = np.array([float(rec[1]), float(rec[2])])

    # 5. 投影4角点世界坐标到图像
    corner_img_proj = _project_world_points(
        H_grid, np.array(list(corner_world.values())))
    if corner_img_proj is None or len(corner_img_proj) != 4:
        print(f"  [角点补全] H_grid 投影失败")
        return None

    corner_types = [1, 2, 3, 4]
    proj_dict = {mt: corner_img_proj[i] for i, mt in enumerate(corner_types)}

    # 6. 用已知角点修正 proj
    if len(known) == 0:
        # 没有任何角点被检测到，直接依赖网格单应性投影四角点
        inferred = {mt: proj_dict[mt] for mt in [1, 2, 3, 4]}
    elif len(known) == 1:
        mt = list(known.keys())[0]
        kp = known[mt]
        best_proj_mt = min(proj_dict.keys(),
                           key=lambda k: float(np.linalg.norm(proj_dict[k] - kp)))
        offset = kp - proj_dict[best_proj_mt]
        inferred = {mt: kp for mt in [1, 2, 3, 4]}
        for mt in [1, 2, 3, 4]:
            if mt not in known:
                inferred[mt] = proj_dict[mt] + offset
    elif len(known) == 2:
        mts = list(known.keys())
        kps = np.array([known[mts[0]], known[mts[1]]])
        proj1, proj2 = proj_dict[mts[0]], proj_dict[mts[1]]
        proj_pts = np.array([proj1, proj2])
        delta_known = kps[1] - kps[0]
        delta_proj = proj_pts[1] - proj_pts[0]
        norm_k, norm_p = float(np.linalg.norm(delta_known)), float(np.linalg.norm(delta_proj))
        if norm_p > 1e-6 and norm_k > 1e-6:
            scale = norm_k / norm_p
            cos_t = float(np.dot(delta_proj, delta_known) / (norm_p * norm_k + 1e-9))
            cos_t = max(-1.0, min(1.0, cos_t))
            sin_t = float(delta_proj[0] * delta_known[1] - delta_proj[1] * delta_known[0]) / (norm_p * norm_k + 1e-9)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            t = kps[0] - scale * R @ proj1
            inferred = {mt: scale * R @ proj_dict[mt] + t for mt in [1, 2, 3, 4]}
        else:
            offset = kps.mean(axis=0) - proj_pts.mean(axis=0)
            inferred = {mt: proj_dict[mt] + offset for mt in [1, 2, 3, 4]}
    elif len(known) == 3:
        mts = list(known.keys())
        proj_pts = np.array([proj_dict[mt] for mt in mts])
        known_pts = np.array([known[mt] for mt in mts])
        A_mat = np.column_stack([proj_pts, np.ones(3)])
        try:
            params_x, _, _, _ = np.linalg.lstsq(A_mat, known_pts[:, 0], rcond=None)
            params_y, _, _, _ = np.linalg.lstsq(A_mat, known_pts[:, 1], rcond=None)
            inferred = {}
            for mt in [1, 2, 3, 4]:
                px, py = proj_dict[mt]
                inferred[mt] = np.array([
                    params_x[0] * px + params_x[1] * py + params_x[2],
                    params_y[0] * px + params_y[1] * py + params_y[2]
                ])
        except Exception:
            offset = known_pts.mean(axis=0) - proj_pts.mean(axis=0)
            inferred = {mt: proj_dict[mt] + offset for mt in [1, 2, 3, 4]}
    else:
        return None

    # 7. 构造4元组jilu返回
    jilu_full = [[mt, float(inferred[mt][0]), float(inferred[mt][1])]
                 for mt in [1, 2, 3, 4]]
    detected_ids = set(known.keys())
    inferred_ids = {1, 2, 3, 4} - detected_ids
    print(f"  [角点补全] 检测到{len(known)}个角点 (types={sorted(detected_ids)}), "
          f"推断{len(inferred_ids)}个缺失角点 (types={sorted(inferred_ids)})")
    return jilu_full


# ===================== 修改点5: 网格拓扑数据输出 =====================
def build_grid_topology(img_pts, world_pts, H, grid_spacing=15.0):
    """
    输出棋盘格的网格拓扑结构数据。
    返回: dict 包含 grid_shape, cells 列表等
    """
    if len(img_pts) < 4 or H is None:
        return None

    snapped = np.round(world_pts / grid_spacing).astype(int)
    cols = np.unique(snapped[:, 0])
    rows = np.unique(snapped[:, 1])
    existing = set(zip(snapped[:, 0], snapped[:, 1]))

    cells = []
    for r in range(len(rows) - 1):
        for c in range(len(cols) - 1):
            gc_tl = (int(cols[c]), int(rows[r]))
            gc_tr = (int(cols[c + 1]), int(rows[r]))
            gc_bl = (int(cols[c]), int(rows[r + 1]))
            gc_br = (int(cols[c + 1]), int(rows[r + 1]))

            has_all = all(g in existing for g in [gc_tl, gc_tr, gc_bl, gc_br])

            cells.append({
                'grid_rc': (r, c),
                'corners_world': {
                    'tl': (float(cols[c] * grid_spacing), float(rows[r] * grid_spacing)),
                    'tr': (float(cols[c + 1] * grid_spacing), float(rows[r] * grid_spacing)),
                    'bl': (float(cols[c] * grid_spacing), float(rows[r + 1] * grid_spacing)),
                    'br': (float(cols[c + 1] * grid_spacing), float(rows[r + 1] * grid_spacing)),
                },
                'has_all_four': has_all
            })

    return {
        'grid_shape': (len(rows), len(cols)),
        'col_range': (int(cols[0]), int(cols[-1])),
        'row_range': (int(rows[0]), int(rows[-1])),
        'n_cells': len(cells),
        'n_complete_cells': sum(1 for c in cells if c['has_all_four']),
        'cells': cells,
    }


def sort_chessboard_points(img_pts, world_pts):
    world_rounded = np.round(world_pts, 2)
    _, unique_idx = np.unique(world_rounded, axis=0, return_index=True)
    img_clean = img_pts[unique_idx]
    world_clean = world_pts[unique_idx]
    sort_idx = np.lexsort([world_clean[:, 0], world_clean[:, 1]])
    return img_clean[sort_idx], world_clean[sort_idx]


def draw_chessboard_grid(ax, img_pts, world_pts, img_shape=None):
    # Rendering uses confirmed image detections.  Do not deduplicate them by
    # world coordinates: a bad post-fill world assignment could otherwise
    # remove a real saddle before geometric row/column grouping.
    img_s = np.asarray(img_pts, dtype=float)
    world_s = np.asarray(world_pts, dtype=float)
    if len(img_s) < 2 or len(world_s) != len(img_s):
        return

    # Dense boards already have reliable integer topology.  Use it directly
    # instead of a single global image direction: under strong perspective,
    # left and right columns converge with different local slopes, so a global
    # angular gate can incorrectly remove an entire edge column.
    grid = np.round(world_s / 15.0).astype(int)
    # BFS/world normalization can occasionally transpose the two lattice axes
    # while preserving an excellent homography (hewei200/204).  Red/blue are
    # physical directions, not arbitrary world-axis names: the more vertical
    # image-space axis must be the red row-increment axis.  Normalize only a
    # genuine 2-D broad grid; narrow/one-column occlusion fallbacks keep their
    # established topology.
    probe_map = {}
    for p, g in zip(img_s, grid):
        probe_map.setdefault((int(g[0]), int(g[1])), np.asarray(p, dtype=float))
    if (len({key[0] for key in probe_map}) >= 3 and
            len({key[1] for key in probe_map}) >= 3):
        x_steps, y_steps = [], []
        for (col, row), p0 in probe_map.items():
            if (col + 1, row) in probe_map:
                x_steps.append(probe_map[(col + 1, row)] - p0)
            if (col, row + 1) in probe_map:
                y_steps.append(probe_map[(col, row + 1)] - p0)

        def median_verticality(vectors):
            if len(vectors) < 4:
                return 0.0
            vectors = np.asarray(vectors, dtype=float)
            lengths = np.linalg.norm(vectors, axis=1)
            valid = lengths > 1e-6
            if np.sum(valid) < 4:
                return 0.0
            return float(np.median(np.abs(vectors[valid, 1]) / lengths[valid]))

        x_verticality = median_verticality(x_steps)
        y_verticality = median_verticality(y_steps)
        if x_verticality > y_verticality + 0.12:
            grid = grid[:, [1, 0]]

    grid_to_img = {}
    for p, g in zip(img_s, grid):
        key = (int(g[0]), int(g[1]))
        if key not in grid_to_img and np.isfinite(p).all():
            grid_to_img[key] = np.asarray(p, dtype=float)
    if grid_to_img:
        cols = [key[0] for key in grid_to_img]
        rows = [key[1] for key in grid_to_img]
        n_cols = max(cols) - min(cols) + 1
        n_rows = max(rows) - min(rows) + 1
        fill_ratio = len(grid_to_img) / max(1, n_cols * n_rows)
    else:
        n_cols = n_rows = 0
        fill_ratio = 0.0

    # A board clipped by the image boundary can legitimately leave only one
    # detected column.  Such a point cloud has no horizontal pair from which
    # estimate_lattice_directions() can recover the second lattice axis, so
    # the generic two-axis renderer below used to draw no red edges at all.
    # Handle only an unambiguous one-column topology here.  Exact consecutive
    # world rows are still required, therefore a missing saddle leaves a gap;
    # interpolation points have already been removed by confirmed_boards.
    if n_cols == 1 and n_rows >= 3 and len(grid_to_img) >= 3:
        centered = img_s - np.median(img_s, axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        line_ratio = (float(singular[0]) /
                      max(float(singular[1]) if len(singular) > 1 else 0.0, 1e-6))
        unique_ratio = len(grid_to_img) / max(len(img_s), 1)
        if line_ratio >= 4.0 and unique_ratio >= 0.80:
            tree = KDTree(img_s)
            nn_dist, _ = tree.query(img_s, k=2)
            one_step = float(np.median(nn_dist[:, 1]))
            drew_edge = False
            for (col, row), p0 in grid_to_img.items():
                p1 = grid_to_img.get((col, row + 1))
                if p1 is None:
                    continue
                dx, dy = p1 - p0
                length = float(np.hypot(dx, dy))
                # Retain the same strict column-like guard used by the sparse
                # multi-column renderer.  The distance gate prevents a
                # compressed/bad row label from bridging a missed grid point.
                if (abs(dy) < 1e-6 or abs(dx) > 0.45 * abs(dy) or
                        length < 0.50 * one_step or length > 1.55 * one_step):
                    continue
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color='#d83b32', linewidth=2, alpha=0.9, zorder=4)
                drew_edge = True
            if drew_edge:
                return

    topology_consistent = False
    # Broad boards can lose several edge/interior saddles and still retain an
    # unambiguous 2-D world topology.  Requiring 90% fill forced hewei179 into
    # the direction-histogram fallback, where numerous diagonal pairs hid the
    # true column mode.  Keep the relaxed fill/aspect allowance restricted to
    # boards with at least four rows and four columns; narrow two-column boards
    # continue to use the stricter safeguards needed for occlusion cases.
    broad_topology = n_cols >= 4 and n_rows >= 4 and fill_ratio >= 0.80
    dense_topology = n_cols >= 2 and n_rows >= 2 and fill_ratio >= 0.90
    if broad_topology or dense_topology:
        dense_H = compute_homography(world_s, img_s)
        if dense_H is not None:
            _, reproj_mean = compute_reprojection_error(dense_H, world_s, img_s)
        else:
            reproj_mean = np.inf
        image_span = np.ptp(img_s, axis=0)
        image_ratio = float(min(image_span) / max(max(image_span), 1e-9))
        world_ratio = float(min(n_cols - 1, n_rows - 1) / max(n_cols - 1, n_rows - 1))
        aspect_factor = max(image_ratio, world_ratio) / max(min(image_ratio, world_ratio), 1e-9)
        axis_separation = 0.0
        if dense_H is not None:
            center_world = np.median(world_s, axis=0)
            probes = np.array([
                center_world,
                center_world + np.array([15.0, 0.0]),
                center_world + np.array([0.0, 15.0]),
            ])
            projected_axes = _project_world_points(dense_H, probes)
            if projected_axes is not None and np.isfinite(projected_axes).all():
                axis_x = projected_axes[1] - projected_axes[0]
                axis_y = projected_axes[2] - projected_axes[0]
                denom = float(np.linalg.norm(axis_x) * np.linalg.norm(axis_y))
                if denom > 1e-9:
                    cos_angle = abs(float(np.dot(axis_x, axis_y))) / denom
                    axis_separation = float(np.degrees(
                        np.arccos(np.clip(cos_angle, 0.0, 1.0))))
        # A falsely grown two-column model can still be nearly full while the
        # actual image point cloud is much wider.  1.9 keeps legitimate strong
        # perspective boards (observed <=1.55) but rejects that topology.
        aspect_limit = 3.0 if broad_topology else 1.9
        normal_topology = reproj_mean <= 4.0 and aspect_factor <= aspect_limit
        # Under severe perspective one physical grid step can be more than
        # three times the other in image space (hewei191/193).  When nearly all
        # world cells are present, reprojection is sub-pixel, and the two axes
        # remain clearly distinct, the world topology is more reliable than
        # the image-direction fallback and must drive red/blue rendering.
        strong_perspective_topology = (
            broad_topology and fill_ratio >= 0.87 and reproj_mean <= 1.5 and
            aspect_factor <= 5.2 and axis_separation >= 35.0)
        topology_consistent = normal_topology or strong_perspective_topology

    if topology_consistent:
        for (col, row), p0 in grid_to_img.items():
            right = grid_to_img.get((col + 1, row))
            if right is not None:
                ax.plot([p0[0], right[0]], [p0[1], right[1]],
                        color='#1f73d8', linewidth=2, alpha=0.9, zorder=4)
            down = grid_to_img.get((col, row + 1))
            if down is not None:
                dx, dy = down - p0
                # A corrupted/locally compressed world label can make two
                # diagonally adjacent saddles look consecutive in one column
                # (hewei207 bottom).  World adjacency remains necessary, but
                # the physical red direction must also be column-like.  If
                # the true next saddle is absent this guard leaves a gap.
                if abs(dy) > 1e-6 and abs(dx) <= 0.45 * abs(dy):
                    ax.plot([p0[0], down[0]], [p0[1], down[1]],
                            color='#d83b32', linewidth=2, alpha=0.9, zorder=4)
        return

    # Estimate the two local lattice axes from confirmed image detections.
    # This avoids corrupted post-fill world labels while retaining the true
    # perspective direction of the board.
    def estimate_lattice_directions():
        """Find the two dominant board axes from all short point pairs."""
        angles = []
        weights = []
        if len(img_s) < 3:
            return None, None
        # Limit pair length relative to the point-cloud nearest-neighbour scale.
        tree = KDTree(img_s)
        nn_dist, _ = tree.query(img_s, k=min(3, len(img_s)))
        base = float(np.median(nn_dist[:, 1])) if nn_dist.ndim == 2 else 20.0
        max_pair = max(30.0, 2.25 * base)
        for i in range(len(img_s)):
            for j in range(i + 1, len(img_s)):
                vec = img_s[j] - img_s[i]
                length = float(np.linalg.norm(vec))
                if length < 6.0 or length > max_pair:
                    continue
                angle = float(np.arctan2(vec[1], vec[0]) % np.pi)
                angles.append(angle)
                weights.append(1.0 / max(length, 1.0))
        if len(angles) < 2:
            return None, None
        n_bins = 90
        hist, edges = np.histogram(angles, bins=n_bins, range=(0.0, np.pi), weights=weights)
        # Circular smoothing prevents sub-pixel angle noise splitting one mode.
        smooth = sum(np.roll(hist, k) for k in range(-2, 3))
        first = int(np.argmax(smooth))
        centers = 0.5 * (edges[:-1] + edges[1:])
        a1 = float(centers[first])
        diff = np.abs(centers - a1)
        diff = np.minimum(diff, np.pi - diff)
        eligible = (diff >= np.deg2rad(50.0)) & (diff <= np.deg2rad(90.0))
        if not np.any(eligible):
            return None, None
        second_scores = np.where(eligible, smooth, -np.inf)
        a2 = float(centers[int(np.argmax(second_scores))])

        def direction(angle):
            v = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            return v

        d1, d2 = direction(a1), direction(a2)
        # Row axis is the one with the larger horizontal component.
        if abs(d1[0]) >= abs(d2[0]):
            row_dir, col_dir = d1, d2
        else:
            row_dir, col_dir = d2, d1
        if row_dir[0] < 0:
            row_dir *= -1.0
        if col_dir[1] < 0:
            col_dir *= -1.0
        return row_dir, col_dir

    row_direction, col_direction = estimate_lattice_directions()

    def estimate_axis(direction):
        if direction is None:
            return None
        samples = []
        normal = np.array([-direction[1], direction[0]])
        angle_tan = np.tan(np.deg2rad(16.0))
        for i, p0 in enumerate(img_s):
            delta = img_s - p0
            distance = np.linalg.norm(delta, axis=1)
            along = np.abs(delta @ direction)
            across = np.abs(delta @ normal)
            mask = (distance > 5.0) & (across <= 2.0 + angle_tan * along)
            ids = np.where(mask)[0]
            if len(ids) == 0:
                continue
            j = int(ids[np.argmin(distance[ids])])
            vec = delta[j].copy()
            if np.dot(vec, direction) < 0:
                vec *= -1.0
            samples.append((float(distance[j]), float(0.5 * (p0[1] + img_s[j, 1]))))
        if not samples:
            return None
        lengths = np.asarray([s[0] for s in samples])
        mids_y = np.asarray([s[1] for s in samples])
        # One-cell image spacing changes with perspective.  Fit it against y
        # so a two-cell jump is rejected even where the board becomes smaller.
        med = float(np.median(lengths))
        good = (lengths >= 0.55 * med) & (lengths <= 1.65 * med)
        if np.sum(good) >= 4 and np.ptp(mids_y[good]) > 10.0:
            spacing_fit = np.polyfit(mids_y[good], lengths[good], 1)
        else:
            spacing_fit = np.array([0.0, med])
        limits = (float(np.percentile(lengths[good], 5)) if np.any(good) else 0.6 * med,
                  float(np.percentile(lengths[good], 95)) if np.any(good) else 1.4 * med)
        return direction, spacing_fit, limits

    row_axis = estimate_axis(row_direction)
    col_axis = estimate_axis(col_direction)

    def connect_axis(axis_model, color, angle_limit_deg=14.0, vertical_edge=False):
        if axis_model is None:
            return
        direction, spacing_fit, limits = axis_model
        normal = np.array([-direction[1], direction[0]])
        angle_tan = np.tan(np.deg2rad(angle_limit_deg))
        drawn = set()
        for i, p0 in enumerate(img_s):
            delta = img_s - p0
            along = delta @ direction
            across = np.abs(delta @ normal)
            predicted = float(np.polyval(spacing_fit, p0[1]))
            predicted = float(np.clip(predicted, 0.75 * limits[0], 1.25 * limits[1]))
            # Only the immediately adjacent cell is eligible.  If that saddle
            # is missing, a two-cell-away point exceeds 1.48 steps and the line
            # remains broken as required.
            mask = ((along > 0.42 * predicted) & (along < 1.48 * predicted) &
                    (across <= 2.0 + angle_tan * along))
            ids = np.where(mask)[0]
            if len(ids) == 0:
                continue
            score = across[ids] + 0.30 * np.abs(along[ids] - predicted)
            j = int(ids[np.argmin(score)])
            p1 = img_s[j]
            dx, dy = p1 - p0
            # A red edge must remain genuinely column-like in image space.
            # This hard guard prevents a bottom point from connecting to a
            # diagonally offset point merely because the global axis drifted.
            if vertical_edge and (dy <= 0.0 or abs(dx) > 0.45 * abs(dy)):
                continue
            key = (min(i, j), max(i, j))
            if key in drawn:
                continue
            drawn.add(key)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=color, linewidth=2, alpha=0.9, zorder=4)

    connect_axis(row_axis, '#1f73d8', angle_limit_deg=14.0, vertical_edge=False)

    def connect_columns_between_rows(row_model, col_model):
        """Match columns one-to-one between consecutive detected rows."""
        if row_model is None or col_model is None:
            return
        row_dir = row_model[0]
        col_dir = col_model[0]
        basis = np.column_stack([row_dir, col_dir])
        if abs(np.linalg.det(basis)) < 1e-4:
            return
        coeff = np.linalg.solve(basis, img_s.T).T
        horizontal_step = float(np.mean(row_model[2]))
        vertical_step = float(np.mean(col_model[2]))

        # Cluster points into physical checkerboard rows using the coordinate
        # along the column basis.  Perspective spreads one row slightly, so
        # the tolerance is a fraction of one vertical cell.
        order = np.argsort(coeff[:, 1])
        row_groups = []
        for idx in order:
            idx = int(idx)
            if not row_groups:
                row_groups.append([idx])
                continue
            center = float(np.median(coeff[row_groups[-1], 1]))
            if abs(float(coeff[idx, 1]) - center) <= 0.36 * vertical_step:
                row_groups[-1].append(idx)
            else:
                row_groups.append([idx])

        row_groups.sort(key=lambda g: float(np.median(coeff[g, 1])))
        for upper, lower in zip(row_groups[:-1], row_groups[1:]):
            upper_level = float(np.median(coeff[upper, 1]))
            lower_level = float(np.median(coeff[lower, 1]))
            # Missing physical row: leave the gap open.
            if lower_level - upper_level > 1.55 * vertical_step:
                continue
            candidates = []
            for i in upper:
                for j in lower:
                    column_error = abs(float(coeff[i, 0] - coeff[j, 0]))
                    if column_error > 0.45 * horizontal_step:
                        continue
                    dx, dy = img_s[j] - img_s[i]
                    if dy <= 0.0 or abs(dx) > 0.45 * abs(dy):
                        continue
                    candidates.append((column_error, int(i), int(j)))
            # Greedy minimum-cost bipartite matching; the strong half-column
            # gate makes assignments unambiguous and prevents column crossing.
            used_upper, used_lower = set(), set()
            for _, i, j in sorted(candidates):
                if i in used_upper or j in used_lower:
                    continue
                used_upper.add(i)
                used_lower.add(j)
                p0, p1 = img_s[i], img_s[j]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color='#d83b32', linewidth=2, alpha=0.9, zorder=4)

    connect_columns_between_rows(row_axis, col_axis)
    return

    # Legacy image-angle grouping retained below for reference only.
    # Some absorbed/interpolated nodes can receive a wrong world row/column.
    # Build the display graph from image geometry instead: nearest neighbours
    # inside a horizontal corridor are blue, and those inside a vertical
    # corridor are red.  The function is called separately for every board,
    # so no edge can cross from one board to another.
    if len(img_s) < 2:
        return
    # Remove sub-pixel duplicates before neighbour search.
    kept = []
    for p in img_s:
        if not kept or min(np.linalg.norm(p - q) for q in kept) > 2.0:
            kept.append(np.asarray(p, dtype=float))
    nodes = np.asarray(kept)
    horizontal = []
    vertical = []
    horizontal_ratio = 0.22  # blue edges must be close to screen-horizontal
    vertical_ratio = 0.38    # red columns may retain moderate perspective tilt
    for i, p0 in enumerate(nodes):
        delta = nodes - p0
        dx, dy = delta[:, 0], delta[:, 1]
        dist = np.hypot(dx, dy)
        hmask = (dx > 2.0) & (np.abs(dy) <= horizontal_ratio * np.abs(dx))
        vmask = (dy > 2.0) & (np.abs(dx) <= vertical_ratio * np.abs(dy))
        if np.any(hmask):
            ids = np.where(hmask)[0]
            j = int(ids[np.argmin(dist[ids])])
            horizontal.append((i, j, float(dist[j])))
        if np.any(vmask):
            ids = np.where(vmask)[0]
            j = int(ids[np.argmin(dist[ids])])
            vertical.append((i, j, float(dist[j])))

    if not horizontal and not vertical:
        return
    h_spacing = float(np.median([e[2] for e in horizontal])) if horizontal else 20.0
    v_spacing = float(np.median([e[2] for e in vertical])) if vertical else h_spacing
    # Do not learn a blue-row slope from candidate edges: one wrong diagonal
    # biases the estimate and then makes row clustering reproduce that error.
    # “蓝横” is explicitly defined in screen/image coordinates here.
    h_slope = 0.0
    v_slope = float(np.median([
        (nodes[j, 0] - nodes[i, 0]) / (nodes[j, 1] - nodes[i, 1])
        for i, j, _ in vertical])) if vertical else 0.0

    def cluster_axis(values, tolerance):
        """Cluster de-sheared 1-D coordinates into stable rows/columns."""
        order = np.argsort(values)
        groups = []
        for idx in order:
            if not groups:
                groups.append([int(idx)])
                continue
            current_center = float(np.median(values[groups[-1]]))
            if abs(float(values[idx]) - current_center) <= tolerance:
                groups[-1].append(int(idx))
            else:
                groups.append([int(idx)])
        return groups

    # Remove the global perspective shear before clustering.  This prevents a
    # nearest-neighbour path from jumping left/right between adjacent columns.
    column_coord = nodes[:, 0] - v_slope * nodes[:, 1]
    row_coord = nodes[:, 1]
    column_groups = cluster_axis(column_coord, max(3.0, 0.38 * h_spacing))
    row_groups = cluster_axis(row_coord, max(3.0, 0.20 * v_spacing))

    def draw_group_edges(groups, sort_axis, expected_slope, typical, color, vertical_edge):
        max_len = 1.75 * typical
        for group in groups:
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda idx: nodes[idx, sort_axis])
            for ia, ib in zip(ordered[:-1], ordered[1:]):
                p0, p1 = nodes[ia], nodes[ib]
                dx, dy = p1 - p0
                length = float(np.hypot(dx, dy))
                if length < 4.0 or length > max_len:
                    continue
                if vertical_edge:
                    # Residual after removing the expected column shear.
                    if dy <= 2.0 or abs(dx - expected_slope * dy) > 0.28 * abs(dy):
                        continue
                else:
                    # Blue edges connect only nearly level saddle points.
                    if dx <= 2.0 or abs(dy) > max(3.0, 0.18 * abs(dx)):
                        continue
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=color, linewidth=2, alpha=0.9, zorder=4)

    draw_group_edges(column_groups, 1, v_slope, v_spacing, '#d83b32', True)
    draw_group_edges(row_groups, 0, h_slope, h_spacing, '#1f73d8', False)
    return

    # Legacy whole-line fitting is retained below for reference but is no
    # longer used; neighbour topology above is the authoritative rendering.
    unique_y = np.unique(np.round(world_s[:, 1], 2))
    unique_x = np.unique(np.round(world_s[:, 0], 2))
    H = compute_homography(world_s, img_s)
    x_min, x_max = float(np.min(unique_x)), float(np.max(unique_x))
    y_min, y_max = float(np.min(unique_y)), float(np.max(unique_y))

    # 图像边界（含一点小 margin，避免数值抖动），用于裁剪网格线
    if img_shape is not None:
        h_img, w_img = float(img_shape[0]), float(img_shape[1])
    else:
        h_img, w_img = None, None

    def clip_segment_to_image(segment):
        """使用 Liang-Barsky 算法将线段裁剪到图像边界内。"""
        if h_img is None or w_img is None:
            return segment
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        dx, dy = x2 - x1, y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1, w_img - x1, y1, h_img - y1]
        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return None
                continue
            r = qi / pi
            if pi < 0:
                u1 = max(u1, r)
            else:
                u2 = min(u2, r)
        if u1 > u2:
            return None
        return np.array([[x1 + u1 * dx, y1 + u1 * dy],
                         [x1 + u2 * dx, y1 + u2 * dy]])

    def project_segment(world_segment):
        if H is None:
            return None
        projected = _project_world_points(H, np.asarray(world_segment, dtype=float))
        if projected is None or not np.isfinite(projected).all():
            return None
        return projected

    def fit_grid_segment(pts, grid_values):
        if len(pts) < 2:
            return None
        order = np.argsort(grid_values)
        pts = pts[order]
        residuals = _point_line_residuals(pts)
        if len(pts) >= 4:
            med = np.median(residuals)
            mad = np.median(np.abs(residuals - med)) + 1e-6
            inliers = residuals <= max(4.0, med + 3.0 * mad)
            pts_fit = pts[inliers] if np.sum(inliers) >= 2 else pts
        else:
            pts_fit = pts
        centered = pts_fit - np.mean(pts_fit, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        t = centered @ direction
        p0 = np.mean(pts_fit, axis=0) + np.min(t) * direction
        p1 = np.mean(pts_fit, axis=0) + np.max(t) * direction

        return np.vstack([p0, p1])

    def line_angle(segment):
        delta = segment[1] - segment[0]
        return np.arctan2(delta[1], delta[0]) % np.pi

    def angle_diff(a, b):
        diff = abs(a - b)
        return min(diff, np.pi - diff)

    def plot_segment(segment):
        span = np.ptp(segment, axis=0)
        color = '#d83b32' if span[1] > span[0] else '#1f73d8'
        clipped = clip_segment_to_image(segment)
        if clipped is not None:
            ax.plot(clipped[:, 0], clipped[:, 1], color=color, linewidth=2, alpha=0.9, zorder=4)

    row_records = []
    col_records = []
    for y in unique_y:
        mask = np.isclose(np.round(world_s[:, 1], 2), y)
        segment = fit_grid_segment(img_s[mask], world_s[mask, 0])
        if segment is not None:
            row_records.append((y, segment))
    for x in unique_x:
        mask = np.isclose(np.round(world_s[:, 0], 2), x)
        segment = fit_grid_segment(img_s[mask], world_s[mask, 1])
        if segment is not None:
            col_records.append((x, segment))

    row_angles = [line_angle(seg) for _, seg in row_records if np.ptp(seg, axis=0)[0] >= np.ptp(seg, axis=0)[1]]
    col_angles = [line_angle(seg) for _, seg in col_records if np.ptp(seg, axis=0)[1] > np.ptp(seg, axis=0)[0]]
    row_med = np.median(row_angles) if row_angles else None
    col_med = np.median(col_angles) if col_angles else None
    angle_limit = np.deg2rad(5.0)
    force_project_rows = False
    force_project_cols = False
    if row_med is not None and row_angles:
        row_spread = np.percentile([angle_diff(a, row_med) for a in row_angles], 90)
        force_project_rows = row_spread > angle_limit
    if col_med is not None and col_angles:
        col_spread = np.percentile([angle_diff(a, col_med) for a in col_angles], 90)
        force_project_cols = col_spread > angle_limit

    for y, segment in row_records:
        use_segment = segment
        if force_project_rows or (row_med is not None and angle_diff(line_angle(segment), row_med) > angle_limit):
            projected = project_segment([[x_min, y], [x_max, y]])
            if projected is not None:
                use_segment = projected
        plot_segment(use_segment)

    for x, segment in col_records:
        use_segment = segment
        if force_project_cols or (col_med is not None and angle_diff(line_angle(segment), col_med) > angle_limit):
            projected = project_segment([[x, y_min], [x, y_max]])
            if projected is not None:
                use_segment = projected
        plot_segment(use_segment)



def plot_chessboard_edge(ax, img_pts, color='red', linewidth=2, linestyle='--'):
    return


def get_chessboard_polygon(img_pts):
    if len(img_pts) < 3:
        return None, None
    try:
        hull = ConvexHull(img_pts)
        hull_pts = img_pts[hull.vertices]
        return Path(hull_pts), hull_pts
    except:
        min_x, min_y = np.min(img_pts, axis=0)
        max_x, max_y = np.max(img_pts, axis=0)
        rect = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        return Path(rect), rect


def find_inside_saddle_points(polygon_path, all_saddle_pts, chess_used_idx):
    if polygon_path is None or len(all_saddle_pts) == 0:
        return np.array([]), np.array([])
    inside = polygon_path.contains_points(all_saddle_pts)
    all_idx = np.arange(len(all_saddle_pts))
    mask = inside & ~np.isin(all_idx, list(chess_used_idx))
    return all_saddle_pts[mask], all_idx[mask]


def _filter_points_inside_polygon(img_pts, world_pts, ref_img_pts, expand_ratio=0.0):
    """
    过滤掉落在 ref_img_pts 凸包外的点。
    用于防止棋盘格重建/插值时把外围候选点误拉进可视化网格。

    expand_ratio: 凸包外扩比例（0=严格不扩展，0.22≈允许约1格边界余量）
    """
    if len(ref_img_pts) < 3 or len(img_pts) == 0:
        return img_pts, world_pts

    try:
        from scipy.spatial import ConvexHull as CH
        hull = CH(ref_img_pts)
        hull_pts = ref_img_pts[hull.vertices]
    except Exception:
        min_xy = np.min(ref_img_pts, axis=0)
        max_xy = np.max(ref_img_pts, axis=0)
        hull_pts = np.array([[min_xy[0], min_xy[1]], [max_xy[0], min_xy[1]],
                             [max_xy[0], max_xy[1]], [min_xy[0], max_xy[1]]])

    if expand_ratio > 0 and len(hull_pts) >= 3:
        centroid = np.mean(hull_pts, axis=0)
        expanded_pts = centroid + (hull_pts - centroid) * (1.0 + expand_ratio)
        poly = Path(expanded_pts)
    else:
        poly = Path(hull_pts)

    inside = poly.contains_points(img_pts)
    if np.sum(inside) < 4:
        return img_pts, world_pts
    return img_pts[inside], world_pts[inside]


def _restore_complete_narrow_end_rows(filtered_img, filtered_world,
                                      completed_img, completed_world,
                                      reference_world, grid_spacing=15.0):
    """Restore one real, complete row just beyond a two-column seed hull.

    Candidate completion works from confirmed saddle detections, but the later
    convex-hull safety filter is intentionally tight.  If a narrow-board seed
    starts at its second row, that filter removes the newly recovered first row
    even when both physical columns were detected (hewei225).  Admit only an
    immediately adjacent end row containing both narrow-board columns.  A
    single response, an interpolated point, or a farther/cross-board row can
    therefore never pass this exception.
    """
    filtered_img = np.asarray(filtered_img, dtype=float)
    filtered_world = np.asarray(filtered_world, dtype=float)
    completed_img = np.asarray(completed_img, dtype=float)
    completed_world = np.asarray(completed_world, dtype=float)
    reference_world = np.asarray(reference_world, dtype=float)
    if (len(reference_world) < 6 or len(completed_img) == 0 or
            len(completed_world) != len(completed_img)):
        return filtered_img, filtered_world, 0

    ref_grid = np.round(reference_world / grid_spacing).astype(int)
    counts = [len(np.unique(ref_grid[:, axis])) for axis in (0, 1)]
    if min(counts) != 2 or max(counts) < 4:
        return filtered_img, filtered_world, 0
    short_axis = int(np.argmin(counts))
    long_axis = 1 - short_axis
    short_values = np.unique(ref_grid[:, short_axis])
    if len(short_values) != 2:
        return filtered_img, filtered_world, 0

    completed_grid = np.round(completed_world / grid_spacing).astype(int)
    long_min = int(np.min(ref_grid[:, long_axis]))
    long_max = int(np.max(ref_grid[:, long_axis]))
    add_indices = []
    for end_value in (long_min - 1, long_max + 1):
        row_indices = []
        for short_value in short_values:
            match = np.where(
                (completed_grid[:, long_axis] == end_value) &
                (completed_grid[:, short_axis] == int(short_value))
            )[0]
            if len(match) != 1:
                row_indices = []
                break
            row_indices.append(int(match[0]))
        # Both columns must contain separately detected candidates.  At this
        # stage interpolation has not run yet, so these are genuine saddles.
        if len(row_indices) == 2:
            add_indices.extend(row_indices)

    if not add_indices:
        return filtered_img, filtered_world, 0
    extra_img = completed_img[np.asarray(add_indices, dtype=int)]
    extra_world = completed_world[np.asarray(add_indices, dtype=int)]
    if len(filtered_img):
        result_img = np.vstack([filtered_img, extra_img])
        result_world = np.vstack([filtered_world, extra_world])
    else:
        result_img, result_world = extra_img, extra_world
    before = len(filtered_img)
    result_img, result_world = _unique_world_points(result_img, result_world)
    return result_img, result_world, max(0, len(result_img) - before)


def _butterfly_contrast_score(img_gray, cx, cy, radius):
    """Return an orientation-independent score for two opposite dark lobes.

    A butterfly mark has two similar, opposite dark sectors and two lighter
    sectors between them.  Ordinary checker intersections and isolated edges
    do not satisfy both the 180-degree symmetry and the angular contrast.
    """
    h, w = img_gray.shape
    radius = float(np.clip(radius, 5.0, 28.0))
    yy, xx = np.mgrid[max(0, int(cy - radius)):min(h, int(cy + radius) + 1),
                      max(0, int(cx - radius)):min(w, int(cx + radius) + 1)]
    if xx.size < 25:
        return 0.0
    dx, dy = xx - cx, yy - cy
    rr = np.hypot(dx, dy)
    annulus = (rr >= 0.28 * radius) & (rr <= radius)
    if np.sum(annulus) < 24:
        return 0.0
    angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    vals = np.asarray(img_gray[yy, xx], dtype=float)
    sector_means = []
    n_sector = 12
    for k in range(n_sector):
        delta = np.angle(np.exp(1j * (angle - 2 * np.pi * k / n_sector)))
        mask = annulus & (np.abs(delta) <= np.pi / n_sector)
        sector_means.append(float(np.mean(vals[mask])) if np.any(mask) else np.nan)
    s = np.asarray(sector_means)
    if not np.all(np.isfinite(s)):
        return 0.0
    dynamic = float(np.percentile(vals[annulus], 90) - np.percentile(vals[annulus], 10))
    # Use the local annulus dynamic range.  Computing whole-image percentiles
    # here made endpoint window search prohibitively expensive because this
    # function is called thousands of times.
    contrast_scale = dynamic + 1e-9
    best = 0.0
    for k in range(n_sector // 2):
        dark = np.array([s[k], s[k + n_sector // 2]])
        side = np.array([s[(k + 3) % n_sector], s[(k + 9) % n_sector]])
        # Both opposite lobes must independently be darker than the side
        # sectors.  Using only their mean allowed a single black edge/lobe to
        # score highly and shifted the reported centre away from the mark.
        side_mean = float(np.mean(side))
        contrast = min(side_mean - float(dark[0]),
                       side_mean - float(dark[1])) / contrast_scale
        symmetry = 1.0 - min(1.0, abs(dark[0] - dark[1]) / (dynamic + 1e-9))
        best = max(best, contrast * max(0.0, symmetry))
    return float(best)


def _butterfly_multiscale_stability(img_gray, cx, cy):
    """Robust two-lobe evidence across marker sizes at one fixed centre."""
    scores = np.asarray([
        _butterfly_contrast_score(img_gray, cx, cy, radius)
        for radius in (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 22.0, 26.0)
    ], dtype=float)
    return float(np.median(scores)), float(np.max(scores))


def _estimate_grid_spacing(img_pts):
    """Robust image-space cell spacing, ignoring duplicate/interpolated nodes."""
    if len(img_pts) < 2:
        return 20.0
    tree = KDTree(img_pts)
    k = min(8, len(img_pts))
    distances, _ = tree.query(img_pts, k=k)
    if distances.ndim == 1:
        return max(8.0, float(np.median(distances)))
    nearest_nonzero = []
    for row in distances:
        valid = row[np.isfinite(row) & (row > 8.0)]
        if len(valid):
            nearest_nonzero.append(float(valid[0]))
    if not nearest_nonzero:
        return 20.0
    return float(np.median(nearest_nonzero))


def _filter_false_positive_x_corners(jilu_b, img_b, img_gray, min_dist_ratio=0.35):
    """
    过滤掉落在棋盘格普通网格角点上的假阳性 X 角点。

    真正的蝴蝶形黑白半圆 X 角点位于棋盘格网格外侧半格处，
    距离最近的棋盘格网格点约为半格间距；而假阳性通常是普通
    棋盘格角点，距离网格点非常近。
    """
    if len(jilu_b) == 0 or len(img_b) < 2:
        return jilu_b

    tree = KDTree(img_b)
    median_spacing = _estimate_grid_spacing(img_b)
    min_dist = max(5.0, min_dist_ratio * median_spacing)
    image_median = float(np.median(img_gray))
    h_img, w_img = img_gray.shape

    filtered = []
    for rec in jilu_b:
        cx, cy = float(rec[1]), float(rec[2])
        d, _ = tree.query([[cx, cy]], k=1)
        appearance = _butterfly_contrast_score(
            img_gray, cx, cy, radius=max(7.0, 0.48 * median_spacing))
        # A valid printed X lies on the light board substrate.  Occluded marks
        # must not be inferred from dark foot/skin texture.
        bg_r = int(round(np.clip(0.65 * median_spacing, 10.0, 35.0)))
        patch = img_gray[max(0, int(cy) - bg_r):min(h_img, int(cy) + bg_r + 1),
                         max(0, int(cx) - bg_r):min(w_img, int(cx) + bg_r + 1)]
        on_bright_board = (patch.size >= 25 and
                           float(np.percentile(patch, 75)) >= image_median)
        # Strong, complete two-lobe evidence is sufficient under cast shadow;
        # endpoint fallback still keeps the bright-board requirement, which
        # rejects occluding foot texture.
        visible_butterfly = appearance >= 0.16 and (on_bright_board or appearance >= 0.80)
        if d[0] >= min_dist and visible_butterfly:
            filtered.append(rec)
        else:
            print(f"  [过滤] 疑似假阳性 X 角点 (type={int(rec[0])}, "
                  f"位置=({cx:.1f},{cy:.1f})): 网格距离={d[0]:.1f}px "
                  f"(阈值{min_dist:.1f}), 蝴蝶外观分={appearance:.3f}, "
                  f"亮底板={on_bright_board}")
    return filtered


def _refine_x_marks_to_butterfly_centers(jilu_b, img_b, img_gray,
                                          roi_radius=82, min_center_score=0.50):
    """Move each detected mark to the midpoint of its two black lobes."""
    if len(jilu_b) == 0 or len(img_b) < 2:
        return jilu_b
    spacing = _estimate_grid_spacing(img_b)
    # Initial template/fallback coordinates may lie on one lobe.  A wider ROI
    # is required to contain both complete half-discs before taking their
    # midpoint.
    # Marker size is not the same as the local saddle spacing (perspective and
    # interpolated duplicates can make that estimate unstable).  Use a fixed
    # maximum ROI large enough for both the small vertical and large
    # horizontal butterfly marks in these images.
    roi_r = int(np.clip(roi_radius, 50, 140))
    h, w = img_gray.shape
    refined = []

    for rec in jilu_b:
        cx, cy = float(rec[1]), float(rec[2])
        x0, x1 = max(0, int(round(cx)) - roi_r), min(w, int(round(cx)) + roi_r + 1)
        y0, y1 = max(0, int(round(cy)) - roi_r), min(h, int(round(cy)) + roi_r + 1)
        patch = np.asarray(img_gray[y0:y1, x0:x1], dtype=float)
        if patch.size < 100:
            refined.append(rec)
            continue

        # Dark lobes occupy a minority of the light substrate.  A percentile
        # threshold is robust to exposure changes and cast shadows.
        p10, p45 = np.percentile(patch, [10, 45])
        dark_threshold = p10 + 0.42 * (p45 - p10)
        mask = patch <= dark_threshold
        labels, count = ndi_label(mask)
        components = []
        patch_area = float(patch.size)
        for lab in range(1, count + 1):
            yy, xx = np.where(labels == lab)
            area = len(xx)
            # Under a cast shadow both black lobes can merge with the dark
            # substrate into one contained component around 30% of this ROI
            # (hewei178/181 upper mark).  Keep it for the merged-waist test;
            # the final angular two-lobe validation below rejects broad dark
            # checker/occluder regions.
            if area < max(28, 0.007 * patch_area) or area > 0.36 * patch_area:
                continue
            # Ignore cropped background/board edges; true lobes are contained.
            if np.any(xx == 0) or np.any(yy == 0) or \
                    np.any(xx == patch.shape[1] - 1) or np.any(yy == patch.shape[0] - 1):
                continue
            components.append({
                'area': float(area),
                'center': np.array([x0 + np.mean(xx), y0 + np.mean(yy)]),
                'mean': float(np.mean(patch[yy, xx])),
                'span': np.array([np.ptp(xx) + 1, np.ptp(yy) + 1], dtype=float),
                'pixels': np.column_stack([x0 + xx, y0 + yy]).astype(float),
            })

        best = None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                a, b = components[i], components[j]
                sep = float(np.linalg.norm(a['center'] - b['center']))
                if sep < 10.0 or sep > 145.0:
                    continue
                area_ratio = min(a['area'], b['area']) / max(a['area'], b['area'])
                if area_ratio < 0.32:
                    continue
                shape_a = float(np.min(a['span']) / max(np.max(a['span']), 1e-9))
                shape_b = float(np.min(b['span']) / max(np.max(b['span']), 1e-9))
                # Oblique views stretch a semicircular lobe to roughly 2:1
                # (observed compactness 0.48 in hewei183--189).  The former
                # 0.50 cut rejected both otherwise clean, balanced components.
                if min(shape_a, shape_b) < 0.34:
                    continue
                midpoint = 0.5 * (a['center'] + b['center'])
                shift = float(np.linalg.norm(midpoint - np.array([cx, cy])))
                if shift > 88.0:
                    continue
                darkness = max(0.0, dark_threshold - 0.5 * (a['mean'] + b['mean']))
                # Prefer two substantial, similarly sized lobes.  The former
                # ratio-dominated score could select two tiny fragments within
                # the upper lobe and therefore place the green point off-centre.
                pair_mass = np.sqrt(a['area'] * b['area'])
                score = pair_mass * area_ratio / (1.0 + shift / (spacing + 1e-9))
                score *= (1.0 + darkness)
                if best is None or score > best[0]:
                    best = (score, midpoint, sep, area_ratio)

        if best is None and components:
            # On small/blurred marks the pointed ends of the two half-discs
            # can touch after thresholding and form one substantial component.
            # Conversely, exposure may split one mark into exactly two large
            # components even when the stricter pair test rejected it.  Use
            # only large, compact components near the initial endpoint prior.
            near = [c for c in components
                    if np.linalg.norm(c['center'] - np.array([cx, cy])) <= 92.0]
            near.sort(key=lambda c: c['area'], reverse=True)
            substantial = [c for c in near
                           if c['area'] >= 0.015 * patch_area and
                           np.min(c['span']) >= 12.0]
            if len(substantial) >= 2:
                a, b = substantial[:2]
                shape_a = float(np.min(a['span']) / max(np.max(a['span']), 1e-9))
                shape_b = float(np.min(b['span']) / max(np.max(b['span']), 1e-9))
                sep = float(np.linalg.norm(a['center'] - b['center']))
                midpoint = 0.5 * (a['center'] + b['center'])
                if (min(shape_a, shape_b) >= 0.34 and
                        10.0 <= sep <= 145.0 and
                        np.linalg.norm(midpoint - np.array([cx, cy])) <= 88.0):
                    ratio = min(a['area'], b['area']) / max(a['area'], b['area'])
                    best = (0.0, midpoint, sep, ratio)
            elif len(substantial) == 1:
                c = substantial[0]
                pix = c['pixels']
                area_fraction = c['area'] / patch_area

                # A one-pixel erosion cleanly separates two lobes whose tips
                # touched during thresholding.  This is more stable than
                # guessing the major axis from a shadow-distorted silhouette
                # (the latter moved hewei182's true centre by over 30 px).
                local_component = np.zeros_like(mask, dtype=bool)
                px = np.round(pix[:, 0] - x0).astype(int)
                py = np.round(pix[:, 1] - y0).astype(int)
                inside = ((px >= 0) & (px < local_component.shape[1]) &
                          (py >= 0) & (py < local_component.shape[0]))
                local_component[py[inside], px[inside]] = True
                # Perspective and blur can make the pointed lobe tips overlap
                # by several pixels.  A single erosion was therefore unable
                # to split the upper butterflies in hewei183--189.  Increase
                # erosion progressively and retain the strongest balanced
                # two-component split; final angular validation below still
                # rejects ordinary checker/edge components.
                split_best = None
                for erosion_iter in range(1, 7):
                    eroded_labels, eroded_count = ndi_label(
                        binary_erosion(local_component, iterations=erosion_iter))
                    eroded_parts = []
                    for er_lab in range(1, eroded_count + 1):
                        ey, ex = np.where(eroded_labels == er_lab)
                        if len(ex) < max(28, 0.0025 * patch_area):
                            continue
                        eroded_parts.append({
                            'area': float(len(ex)),
                            'center': np.array([x0 + np.mean(ex), y0 + np.mean(ey)])
                        })
                    eroded_parts.sort(key=lambda part: part['area'], reverse=True)
                    if len(eroded_parts) < 2 or area_fraction < 0.045:
                        continue
                    a, b = eroded_parts[:2]
                    sep = float(np.linalg.norm(a['center'] - b['center']))
                    ratio = min(a['area'], b['area']) / max(a['area'], b['area'])
                    midpoint = 0.5 * (a['center'] + b['center'])
                    shift = float(np.linalg.norm(midpoint - np.array([cx, cy])))
                    if (10.0 <= sep <= 145.0 and ratio >= 0.25 and shift <= 92.0):
                        split_score = np.sqrt(a['area'] * b['area']) * ratio
                        if split_best is None or split_score > split_best[0]:
                            split_best = (split_score, midpoint, sep, ratio)
                if split_best is not None:
                    best = split_best

                # Small printed butterflies often remain one compact component
                # even after erosion.  Their bounding-box midpoint is more
                # stable than a darkness centroid.  The narrow area interval
                # excludes checker runs; the angular centre score still has
                # the final say.
                if best is None and 0.015 <= area_fraction <= 0.035:
                    midpoint = 0.5 * (np.min(pix, axis=0) + np.max(pix, axis=0))
                    if np.linalg.norm(midpoint - np.array([cx, cy])) <= 92.0:
                        best = (0.0, midpoint, float(np.max(c['span'])), 1.0)

        if best is not None:
            midpoint = best[1]
            appearance_radius = float(np.clip(0.28 * best[2], 14.0, 28.0))
            center_appearance = _butterfly_contrast_score(
                img_gray, midpoint[0], midpoint[1], appearance_radius)
            # Revalidate at the refined centre.  A vertical run of checker
            # squares can have a narrow connected-component waist, but it
            # lacks two opposite dark lobes around that waist (hewei181).
            if center_appearance < min_center_score:
                raw_scores = np.asarray([
                    _butterfly_contrast_score(img_gray, cx, cy, radius)
                    for radius in (8.0, 10.0, 12.0, 14.0,
                                   16.0, 18.0, 22.0, 26.0)
                ], dtype=float)
                raw_appearance = float(np.max(raw_scores))
                raw_stability = float(np.median(raw_scores))
                # A large ROI can select distant checker components and pull
                # the midpoint away from an already excellent small-marker
                # centre (hewei259).  Preserve the raw centre only when its
                # evidence is decisively stronger; common filters still run.
                if (raw_appearance >= 0.85 and raw_stability >= 0.65 and
                        raw_appearance >= center_appearance + 0.30):
                    refined.append([int(rec[0]), float(cx), float(cy)])
                    print(f"  [X小标记保留原中心] 中心=({cx:.1f},{cy:.1f}), "
                          f"原始分={raw_appearance:.3f}, "
                          f"多尺度中位分={raw_stability:.3f}, "
                          f"偏移中心分={center_appearance:.3f}")
                    continue
                print(f"  [X过滤] ({midpoint[0]:.1f},{midpoint[1]:.1f}) "
                      f"中心双瓣外观分={center_appearance:.3f}，不标出")
                continue
            print(f"  [X中心校正] ({cx:.1f},{cy:.1f}) -> "
                  f"({midpoint[0]:.1f},{midpoint[1]:.1f}), "
                  f"双瓣间距={best[2]:.1f}px, 面积比={best[3]:.2f}, "
                  f"中心外观分={center_appearance:.3f}")
            refined.append([int(rec[0]), float(midpoint[0]), float(midpoint[1])])
        else:
            # A printed mark clipped by the image boundary can lose one
            # connected-component lobe even though its predicted endpoint
            # centre still shows two opposite dark sectors (hewei250).  This
            # is image clipping, not an occluded mark; retain only a very
            # strong boundary-centre response.
            near_image_boundary = (
                cx <= 0.06 * w or cx >= 0.94 * w or
                cy <= 0.04 * h or cy >= 0.96 * h)
            boundary_score = 0.0
            if near_image_boundary:
                boundary_score = max(_butterfly_contrast_score(
                    img_gray, cx, cy, radius)
                    for radius in (14.0, 18.0, 22.0, 26.0, 30.0))
            if near_image_boundary and boundary_score >= 0.80:
                refined.append([int(rec[0]), float(cx), float(cy)])
                print(f"  [X边界裁切保留] 中心=({cx:.1f},{cy:.1f}), "
                      f"双瓣外观分={boundary_score:.3f}")
                continue
            # No complete pair means the mark is clipped/occluded; do not
            # display a point guessed from a single visible dark region.
            print(f"  [X过滤] ({cx:.1f},{cy:.1f}) 未找到完整双瓣，不标出")
    return refined


def _refine_x_marks_multiscale(jilu_b, img_b, img_gray,
                               roi_radii=(125, 100, 82),
                               min_center_score=0.50):
    """Refine large and small butterflies without changing successful marks."""
    refined = []
    for rec in jilu_b:
        result = []
        for roi_radius in roi_radii:
            result = _refine_x_marks_to_butterfly_centers(
                [rec], img_b, img_gray, roi_radius=roi_radius,
                min_center_score=min_center_score)
            if result:
                break
        refined.extend(result)
    return refined


def _detect_endpoint_butterfly_fallback(img_b, world_b, H_b, img_gray,
                                         existing, grid_spacing=15.0):
    """Find butterfly marks beyond the ends of a narrow chessboard strip.

    Unlike ``get_mark_cord`` this fallback does not require the mark centre to
    have survived the checkerboard saddle templates.  Its search region is
    deliberately restricted to the projected centre line, 1.2--3.6 cells
    beyond either end of a two-column strip.
    """
    if H_b is None or len(img_b) < 6 or len(world_b) < 6:
        return []
    snapped = np.round(world_b / grid_spacing).astype(int)
    ux, uy = np.unique(snapped[:, 0]), np.unique(snapped[:, 1])
    # The physical target uses a narrow two-column saddle strip.  Do not run
    # the fallback on broad/irregular boards, where endpoint priors are weak.
    if len(ux) != 2 or len(uy) < 4:
        return []

    x_mid = 0.5 * (float(np.min(ux)) + float(np.max(ux))) * grid_spacing
    y_min, y_max = float(np.min(uy)) * grid_spacing, float(np.max(uy)) * grid_spacing
    anchors_w = np.array([[x_mid, y_min], [x_mid, y_max]], dtype=float)
    inner_w = np.array([[x_mid, y_min + grid_spacing],
                        [x_mid, y_max - grid_spacing]], dtype=float)
    anchors = _project_world_points(H_b, anchors_w)
    inner = _project_world_points(H_b, inner_w)
    if anchors is None or inner is None or not np.isfinite(anchors).all():
        return []

    h_img, w_img = img_gray.shape
    image_median = float(np.median(img_gray))
    existing_xy = np.asarray([[float(r[1]), float(r[2])] for r in existing], dtype=float) \
        if len(existing) else np.empty((0, 2))
    found = []
    for end_idx in range(2):
        outward = anchors[end_idx] - inner[end_idx]
        cell_px = float(np.linalg.norm(outward))
        if cell_px < 8.0:
            continue
        outward /= cell_px
        lateral = np.array([-outward[1], outward[0]])
        # If normal template detection already found a mark beyond this exact
        # strip end, do not search the near checker cells again.  The previous
        # distance-only dedup allowed hewei225 to add a second green point on
        # the first black square below the genuine upper-right butterfly.
        if len(existing_xy):
            relative = existing_xy - anchors[end_idx]
            along_existing = relative @ outward
            lateral_existing = np.abs(relative @ lateral)
            endpoint_known = np.any(
                (along_existing >= 0.45 * cell_px) &
                (along_existing <= 8.0 * cell_px) &
                (lateral_existing <= 2.0 * cell_px))
            if endpoint_known:
                continue
        radius = float(np.clip(0.58 * cell_px, 14.0, 30.0))
        best_score, best_pos = 0.0, None
        # Coarse pixel search.  Longitudinal restriction is what prevents
        # checker intersections inside the strip from becoming candidates.
        for along in np.arange(1.2 * cell_px, 3.61 * cell_px, 3.0):
            base = anchors[end_idx] + along * outward
            for side in np.arange(-0.48 * cell_px, 0.481 * cell_px, 3.0):
                p = base + side * lateral
                cx, cy = float(p[0]), float(p[1])
                if cx < radius or cy < radius or cx >= w_img - radius or cy >= h_img - radius:
                    continue
                score = _butterfly_contrast_score(img_gray, cx, cy, radius)
                if score > best_score:
                    best_score, best_pos = score, (cx, cy)
        # The endpoint prior is strong, so this threshold can be lower than
        # the generic full-image appearance threshold.
        # Endpoint fallback has a strong geometric prior but must still show a
        # complete butterfly.  A lower threshold admitted checker squares at
        # cropped image borders (observed scores 0.15--0.45).
        if best_pos is None or best_score < 0.55:
            continue
        # The mark is printed on the bright board substrate.  This rejects
        # symmetric texture on the dark/occluding foot beyond the lower end.
        bx, by = best_pos
        r_bg = int(round(1.35 * radius))
        bg_patch = img_gray[max(0, int(by) - r_bg):min(h_img, int(by) + r_bg + 1),
                            max(0, int(bx) - r_bg):min(w_img, int(bx) + r_bg + 1)]
        if bg_patch.size < 25 or float(np.percentile(bg_patch, 75)) < image_median:
            continue

        # The angular score can peak near a lobe edge.  Re-centre on the dark
        # mass in a larger patch so the plotted point lies in the white gap
        # between the two black half-discs.
        # The coarse angular maximum can sit roughly one lobe radius outside
        # the true centre, so include both lobes with a wider refinement ROI.
        r_ref = int(round(3.2 * radius))
        x0, x1 = max(0, int(bx) - r_ref), min(w_img, int(bx) + r_ref + 1)
        y0, y1 = max(0, int(by) - r_ref), min(h_img, int(by) + r_ref + 1)
        patch = np.asarray(img_gray[y0:y1, x0:x1], dtype=float)
        dark_cut = float(np.percentile(patch, 32))
        weights = np.clip(dark_cut - patch, 0.0, None)
        if np.sum(weights) > 1e-6:
            gy, gx = np.mgrid[y0:y1, x0:x1]
            p = np.array([np.sum(gx * weights) / np.sum(weights),
                          np.sum(gy * weights) / np.sum(weights)])
        else:
            p = np.asarray(best_pos)
        if len(existing_xy) and np.min(np.linalg.norm(existing_xy - p, axis=1)) < 0.65 * cell_px:
            continue
        if found and np.min([np.linalg.norm(np.asarray(r[1:3]) - p) for r in found]) < 0.65 * cell_px:
            continue
        # Stable geometric type for reporting: 1/2 upper-left/right,
        # 3/4 lower-left/right in the image.
        ctype = (1 if p[0] < w_img / 2 else 2) if p[1] < h_img / 2 else \
                (3 if p[0] < w_img / 2 else 4)
        found.append([ctype, float(p[0]), float(p[1])])
        print(f"  [端点回退] 检测到蝴蝶X角点 type={ctype}, "
              f"位置=({p[0]:.1f},{p[1]:.1f}), 外观分={best_score:.3f}")
    return found


def _detect_butterfly_above_sparse_strip(candidate_pts, img_gray,
                                           existing_boards, existing_marks):
    """Recover an X above a severely occluded, not-yet-grown narrow strip.

    The fallback is not a global X detector.  A candidate must show butterfly
    contrast and must have two vertically aligned genuine saddle responses
    below it, with the first saddle more than one row step away.  Thus an
    ordinary checker intersection cannot validate itself as an X.
    """
    if len(candidate_pts) < 3:
        return []
    available = _available_strip_points_mask(candidate_pts, existing_boards)
    seed_ids = np.where(available)[0]
    if len(seed_ids) == 0:
        return []
    existing_xy = np.asarray([m[1:3] for m in existing_marks], dtype=float) \
        if len(existing_marks) else np.empty((0, 2))
    image_median = float(np.median(img_gray))
    h, w = img_gray.shape
    existing_slots = {
        (0 if float(mark[1]) < 0.5 * w else 1,
         0 if float(mark[2]) < 0.5 * h else 1)
        for mark in existing_marks
    }
    recovered = []

    for idx in seed_ids:
        seed = np.asarray(candidate_pts[int(idx)], dtype=float)
        # This fallback is only for the two narrow strips on the central
        # target.  Cropped intersections of the broad checkerboard along the
        # extreme left edge can also show two-lobe contrast after perspective
        # shape relaxation (hewei175), but can never be an X target here.
        if seed[0] < 0.18 * w or seed[0] > 0.92 * w:
            continue
        seed_slot = (0 if seed[0] < 0.5 * w else 1,
                     0 if seed[1] < 0.5 * h else 1)
        if seed_slot in existing_slots:
            continue
        if _is_broad_board_lattice_extension(seed, existing_boards):
            continue
        if len(existing_xy) and np.min(np.linalg.norm(existing_xy - seed, axis=1)) < 45.0:
            continue
        appearance = max(_butterfly_contrast_score(
            img_gray, seed[0], seed[1], radius) for radius in (14.0, 18.0, 22.0, 26.0))
        if appearance < 0.70:
            continue
        bg_r = 28
        patch = img_gray[max(0, int(seed[1]) - bg_r):min(h, int(seed[1]) + bg_r + 1),
                         max(0, int(seed[0]) - bg_r):min(w, int(seed[0]) + bg_r + 1)]
        if patch.size < 25 or float(np.percentile(patch, 75)) < image_median:
            continue

        delta = np.asarray(candidate_pts, dtype=float) - seed
        below = np.where((delta[:, 1] >= 70.0) & (delta[:, 1] <= 430.0) &
                         (np.abs(delta[:, 0]) <= 65.0))[0]
        if len(below) < 2:
            continue
        support = np.asarray(candidate_pts[below], dtype=float)
        order = np.argsort(support[:, 1])
        support = support[order]
        best_pair = None
        for i in range(len(support)):
            for j in range(i + 1, len(support)):
                row_step = float(support[j, 1] - support[i, 1])
                if row_step < 35.0 or row_step > 130.0:
                    continue
                if abs(float(support[j, 0] - support[i, 0])) > 20.0:
                    continue
                first_gap = float(support[i, 1] - seed[1])
                if first_gap < 1.35 * row_step or first_gap > 5.5 * row_step:
                    continue
                lateral = abs(float(0.5 * (support[i, 0] + support[j, 0]) - seed[0]))
                # The printed X is centred over the narrow strip.  Candidates
                # tens of pixels to one side are usually plastic/board edges
                # whose large ROI later drifts onto a checker square.
                if lateral > 25.0:
                    continue
                score = appearance - 0.006 * lateral - 0.001 * first_gap
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, support[i], support[j])
        if best_pair is None:
            continue

        ctype = (1 if seed[0] < w / 2 else 2) if seed[1] < h / 2 else \
                (3 if seed[0] < w / 2 else 4)
        strip_evidence = np.asarray([best_pair[1], best_pair[2]], dtype=float)
        refined = _refine_x_marks_to_butterfly_centers(
            [[ctype, float(seed[0]), float(seed[1])]], strip_evidence,
            img_gray, roi_radius=82, min_center_score=0.50)
        refined = _filter_false_positive_x_corners(
            refined, strip_evidence, img_gray, min_dist_ratio=0.35)
        for rec in refined:
            p = np.asarray(rec[1:3], dtype=float)
            rec_slot = (0 if p[0] < 0.5 * w else 1,
                        0 if p[1] < 0.5 * h else 1)
            if rec_slot in existing_slots:
                continue
            known = list(existing_marks) + recovered
            if known and min(np.linalg.norm(p - np.asarray(old[1:3], dtype=float))
                             for old in known) < 35.0:
                continue
            recovered.append(rec)
            print(f"  [稀疏条带X回退] 位置=({p[0]:.1f},{p[1]:.1f}), "
                  f"蝴蝶分={appearance:.3f}")
    return recovered


def _detect_isolated_complete_butterflies(candidate_pts, img_gray,
                                            existing_boards, existing_marks):
    """Recover visible upper butterflies when the strip below is occluded.

    This is deliberately not a general image-wide X inference.  Seeds must be
    strong saddle candidates outside every accepted board, lie on the bright
    central target substrate, and pass the same complete two-lobe refinement
    and grid-distance rejection as board-guided marks.  At most the two upper
    strip endpoints used by this physical target are retained.
    """
    if len(candidate_pts) == 0:
        return []
    points = np.asarray(candidate_pts, dtype=float)
    available = _available_strip_points_mask(points, existing_boards, margin=14.0)
    h, w = img_gray.shape
    image_median = float(np.median(img_gray))
    known_upper_sides = {
        0 if float(mark[1]) < 0.5 * w else 1
        for mark in existing_marks if float(mark[2]) <= 0.58 * h
    }
    if len(known_upper_sides) >= 2:
        return []
    seeds = []
    for idx in np.where(available)[0]:
        p = points[int(idx)]
        # The isolated fallback is needed only at the two upper endpoints.
        # The x bound excludes the wide checkerboard/reflection along the left
        # image edge without constraining the endpoint orientation.
        if not (0.18 * w <= p[0] <= 0.92 * w and
                0.04 * h <= p[1] <= 0.58 * h):
            continue
        if (_is_broad_board_lattice_extension(p, existing_boards) or
                _is_near_broad_board_image_region(p, existing_boards)):
            continue
        appearance = max(_butterfly_contrast_score(
            img_gray, p[0], p[1], radius) for radius in (14.0, 18.0, 22.0, 26.0))
        if appearance < 0.80:
            continue
        bg_r = 30
        patch = img_gray[max(0, int(p[1]) - bg_r):min(h, int(p[1]) + bg_r + 1),
                         max(0, int(p[0]) - bg_r):min(w, int(p[0]) + bg_r + 1)]
        if patch.size < 25 or float(np.percentile(patch, 75)) < image_median:
            continue
        seeds.append((appearance, p))

    # A complete butterfly centre is not guaranteed to produce a saddle
    # response.  In hewei203 both dark lobes are clear, but no final_points
    # seed survives at their white centre, so the candidate-seeded fallback
    # above never starts.  Search only a missing upper physical side and only
    # outside broad-board bounding boxes; the high appearance threshold and
    # the existing complete-lobe refinement remain mandatory.
    broad_boxes = []
    has_narrow_board = False
    for board in existing_boards:
        board_img = np.asarray(board[0], dtype=float)
        board_world = np.asarray(board[1], dtype=float)
        if len(board_img) < 4 or len(board_world) != len(board_img):
            continue
        board_grid = np.round(board_world / 15.0).astype(int)
        grid_counts = (len(np.unique(board_grid[:, 0])),
                       len(np.unique(board_grid[:, 1])))
        if min(grid_counts) <= 2 and max(grid_counts) >= 4:
            has_narrow_board = True
        if len(board_img) < 12:
            continue
        if (len(np.unique(board_grid[:, 0])) < 4 or
                len(np.unique(board_grid[:, 1])) < 4):
            continue
        broad_boxes.append((
            float(np.min(board_img[:, 0]) - 24.0),
            float(np.max(board_img[:, 0]) + 24.0),
            float(np.min(board_img[:, 1]) - 24.0),
            float(np.max(board_img[:, 1]) + 24.0),
        ))

    for side in (0, 1):
        # Existing candidate seeds may be strong checker/edge responses that
        # later fail complete-lobe refinement.  They must not suppress the
        # direct search for a genuinely missing physical side (hewei203).
        if (side in known_upper_sides or
                (not existing_marks and not has_narrow_board)):
            continue
        if side == 0:
            x_start, x_stop = int(0.25 * w), int(0.56 * w)
        else:
            # Strong perspective can move the physical right endpoint close
            # to the image centre (hewei259 small upper-right butterfly).
            x_start, x_stop = int(0.46 * w), int(0.91 * w)
        y_start, y_stop = int(0.10 * h), int(0.42 * h)
        coarse_candidates = []
        for cy in range(y_start, y_stop + 1, 8):
            for cx in range(x_start, x_stop + 1, 8):
                if any(x0 <= cx <= x1 and y0 <= cy <= y1
                       for x0, x1, y0, y1 in broad_boxes):
                    continue
                score = max(_butterfly_contrast_score(
                    img_gray, cx, cy, radius) for radius in (18.0, 24.0))
                if score >= 0.75:
                    coarse_candidates.append(
                        (score, np.array([float(cx), float(cy)])))
        if not coarse_candidates:
            continue

        # Retain several spatial peaks.  The absolute angular maximum may sit
        # on a nearby symmetric edge; complete-lobe centring is authoritative.
        coarse_kept = []
        for score, point in sorted(coarse_candidates,
                                   key=lambda item: item[0], reverse=True):
            if (coarse_kept and min(np.linalg.norm(point - old[1])
                                    for old in coarse_kept) < 24.0):
                continue
            coarse_kept.append((score, point))
            if len(coarse_kept) >= 8:
                break

        for coarse_score, coarse in coarse_kept:
            best_score, best_point = float(coarse_score), coarse.copy()
            for cy in np.arange(coarse[1] - 8.0, coarse[1] + 8.1, 2.0):
                for cx in np.arange(coarse[0] - 8.0, coarse[0] + 8.1, 2.0):
                    score = max(_butterfly_contrast_score(
                        img_gray, cx, cy, radius)
                        for radius in (14.0, 18.0, 22.0, 26.0))
                    if score > best_score:
                        best_score = score
                        best_point = np.array([float(cx), float(cy)])
            if best_score < 0.95:
                continue
            if (_is_broad_board_lattice_extension(
                    best_point, existing_boards) or
                    _is_near_broad_board_image_region(
                        best_point, existing_boards)):
                continue
            bg_r = 30
            bx, by = best_point
            bg_patch = img_gray[
                max(0, int(by) - bg_r):min(h, int(by) + bg_r + 1),
                max(0, int(bx) - bg_r):min(w, int(bx) + bg_r + 1)]
            # The central-left substrate can be slightly darker than the
            # global median under the foot's cast shadow (hewei203).  A very
            # strong score may proceed to the stricter complete-lobe test.
            if (bg_patch.size < 25 or
                    (float(np.percentile(bg_patch, 75)) < image_median and
                     best_score < 1.10)):
                continue
            raw_type = 1 if side == 0 else 2
            trial = _refine_x_marks_multiscale(
                [[raw_type, float(bx), float(by)]], points, img_gray,
                min_center_score=0.50)
            if not trial:
                continue
            centred = np.asarray(trial[0][1:3], dtype=float)
            if (_is_broad_board_lattice_extension(
                    centred, existing_boards) or
                    _is_near_broad_board_image_region(
                        centred, existing_boards)):
                continue
            # Give the proven centre priority over nearby unrefined saddle
            # seeds during the common NMS below.
            seeds.append((10.0 + best_score, centred))
            print(f"  [完整蝴蝶直接搜索] 候选=({bx:.1f},{by:.1f}) -> "
                  f"中心=({centred[0]:.1f},{centred[1]:.1f}), "
                  f"外观分={best_score:.3f}")
            break
    if not seeds:
        return []

    # Spatial NMS: several corner-template responses can sit on one butterfly.
    kept = []
    for appearance, p in sorted(seeds, key=lambda item: item[0], reverse=True):
        if kept and min(np.linalg.norm(p - old[1]) for old in kept) < 48.0:
            continue
        kept.append((appearance, p))

    if existing_boards:
        board_img = np.vstack([np.asarray(board[0], dtype=float)
                               for board in existing_boards if len(board[0])])
    else:
        board_img = np.empty((0, 2), dtype=float)
    support_img = board_img if len(board_img) >= 2 else points
    raw = []
    for _, p in kept:
        ctype = 1 if p[0] < w / 2 else 2
        raw.append([ctype, float(p[0]), float(p[1])])
    refined = _refine_x_marks_multiscale(
        raw, support_img, img_gray, min_center_score=0.50)
    refined = _filter_false_positive_x_corners(
        refined, support_img, img_gray, min_dist_ratio=0.35)
    clean_refined = []
    for rec in refined:
        p = np.asarray(rec[1:3], dtype=float)
        if (_is_broad_board_lattice_extension(p, existing_boards) or
                _is_near_broad_board_image_region(p, existing_boards)):
            print(f"  [X宽棋盘排除] 位置=({p[0]:.1f},{p[1]:.1f})")
            continue
        clean_refined.append(rec)
    refined = clean_refined

    # A checker square can produce a high response at one favourable radius,
    # whereas a true butterfly remains two-lobed over a range of radii.  Rank
    # the final *centred* candidates by that stability, and reject unstable
    # checker-like centres before assigning the one available physical side.
    ranked_refined = []
    for rec in refined:
        stability, peak = _butterfly_multiscale_stability(
            img_gray, float(rec[1]), float(rec[2]))
        if stability < 0.74:
            print(f"  [X棋格排除] 位置=({rec[1]:.1f},{rec[2]:.1f}), "
                  f"多尺度中位分={stability:.3f}, 峰值={peak:.3f}")
            continue
        ranked_refined.append((stability, peak, rec))
    ranked_refined.sort(key=lambda item: (item[0], item[1]), reverse=True)

    known = [np.asarray(mark[1:3], dtype=float) for mark in existing_marks]
    top_slots = max(0, 2 - len(known_upper_sides))
    if top_slots <= 0:
        return []
    recovered = []
    recovered_sides = set()
    for _, _, rec in ranked_refined:
        p = np.asarray(rec[1:3], dtype=float)
        side = 0 if p[0] < 0.5 * w else 1
        if side in known_upper_sides or side in recovered_sides:
            continue
        if known and min(np.linalg.norm(p - old) for old in known) < 45.0:
            continue
        if recovered and min(np.linalg.norm(
                p - np.asarray(old[1:3], dtype=float)) for old in recovered) < 45.0:
            continue
        recovered.append(rec)
        recovered_sides.add(side)
        if len(recovered) >= top_slots:
            break
    for rec in recovered:
        print(f"  [完整蝴蝶回退] 位置=({rec[1]:.1f},{rec[2]:.1f})")
    return recovered


def _normalize_physical_x_corner_types(records, img_shape):
    """Map detected image positions to the target's physical corner numbers.

    The physical definition used by this target is clockwise when viewed in
    the normalized image: upper-left=4, upper-right=3, lower-left=1 and
    lower-right=2.  Individual legacy detectors use different local numbering
    conventions, so their raw type must not be used directly for display or
    downstream results.
    """
    h, w = img_shape[:2]
    normalized = []
    for rec in records:
        cx, cy = float(rec[1]), float(rec[2])
        if cy < 0.5 * h:
            mark_type = 4 if cx < 0.5 * w else 3
        else:
            mark_type = 1 if cx < 0.5 * w else 2
        normalized.append([mark_type, cx, cy])
    return normalized


def validate_chessboard_grid_alignment(world_pts, max_deviation=1.5):
    if len(world_pts) < 4:
        return False, 1e9
    snapped = np.round(world_pts / 15.0) * 15.0
    dev = np.linalg.norm(world_pts - snapped, axis=1)
    median_dev = np.median(dev)
    if np.max(dev) > max_deviation * 1.5:
        return False, median_dev
    if median_dev > max_deviation:
        return False, median_dev
    ux = np.unique(snapped[:, 0])
    uy = np.unique(snapped[:, 1])
    if len(ux) < 2 or len(uy) < 2:
        return False, median_dev
    return True, median_dev



def calculate_chessboard_quality(img_pts, world_pts):
    if len(img_pts) < 4:
        return 0
    H = compute_homography(world_pts, img_pts)
    if H is None:
        return 0
    _, mean_e = compute_reprojection_error(H, world_pts, img_pts)
    p90_line, _, bad_groups = chessboard_line_residual_summary(img_pts, world_pts)
    snapped = np.round(world_pts / 15.0).astype(int)
    grid_cells = len(np.unique(snapped[:, 0])) * len(np.unique(snapped[:, 1]))
    fill_ratio = len(img_pts) / max(1, grid_cells)
    fill_ratio = float(np.clip(fill_ratio, 0.0, 1.0))

    reproj_penalty = np.exp(-0.5 * (mean_e / 3.0) ** 2)
    line_penalty = np.exp(-0.5 * (p90_line / 4.0) ** 2)
    bad_group_penalty = 1.0 / (1.0 + bad_groups)
    score = len(img_pts) * (0.55 + 0.45 * fill_ratio)
    score *= reproj_penalty * line_penalty * bad_group_penalty
    return float(score)


def prepare_chessboard_candidate(img_pts, world_pts, candidate_pts, H=None):
    clean_img, clean_world, clean_H = filter_chessboard_line_outliers(
        img_pts, world_pts, threshold=7.5
    )
    p90_line, _, bad_groups = chessboard_line_residual_summary(clean_img, clean_world)
    if len(clean_img) < 120 and p90_line < 2.0 and bad_groups == 0:
        adj_img, adj_world, adj_H = adjust_skipped_grid_axis(
            clean_img, clean_world, candidate_pts, min_inside=3
        )
        adj_p90, _, adj_bad = chessboard_line_residual_summary(adj_img, adj_world)
        old_quality = calculate_chessboard_quality(clean_img, clean_world)
        new_quality = calculate_chessboard_quality(adj_img, adj_world)
        if adj_bad == 0 and adj_p90 <= max(2.0, p90_line + 0.5) and new_quality >= old_quality * 0.65:
            return adj_img, adj_world, adj_H if adj_H is not None else clean_H
    return clean_img, clean_world, clean_H if clean_H is not None else H


# ===================== 单应矩阵传播：推算第二个棋盘格 =====================
def propagate_chessboard_by_homography(first_img_pts, first_world_pts, first_H,
                                        candidate_pts, grid_spacing=15.0,
                                        min_points=6):
    """
    利用已识别棋盘格的单应矩阵，正向投影预测第二棋盘格的位置，再匹配候选点。

    策略：不依赖反向投影+网格对齐（外推误差大），而是：
    1. 在第一棋盘格世界网格的四周扩展搜索窗口
    2. 用 H 将扩展网格点投影到图像空间
    3. KDTree 搜索每个投影点附近的候选点
    4. 若命中数足够，构建第二棋盘格

    :param first_img_pts:  第一个棋盘格的图像坐标 (N,2)
    :param first_world_pts: 第一个棋盘格的世界坐标 (N,2)，已归零
    :param first_H:        第一个棋盘格的单应矩阵 (3,3) world->image
    :param candidate_pts:  所有鞍点图像坐标 (M,2)
    :param grid_spacing:   棋盘格单元间距 mm
    :param min_points:     最少需要的网格点数
    :return: (img_pts, world_pts, H, used_indices) 或 None
    """
    if first_H is None or len(candidate_pts) < min_points:
        return None

    # 1. 解析第一棋盘格的网格行列范围
    snapped1 = np.round(first_world_pts / grid_spacing).astype(int)
    col_min, col_max = int(np.min(snapped1[:, 0])), int(np.max(snapped1[:, 0]))
    row_min, row_max = int(np.min(snapped1[:, 1])), int(np.max(snapped1[:, 1]))
    n_cols1 = col_max - col_min + 1
    n_rows1 = row_max - row_min + 1

    print(f"  [传播] 第一棋盘: {n_cols1}列×{n_rows1}行, "
          f"范围 col[{col_min},{col_max}] row[{row_min},{row_max}]")

    # 2. 构建候选点 KDTree
    tree = KDTree(candidate_pts)

    # 3. 四方向扩展搜索（+X, -X, +Y, -Y），每种方向多个间隙
    best_result = None

    for axis_dir, axis_name in [(0, "X"), (1, "Y")]:
        for sign, sign_name in [(+1, "+"), (-1, "-")]:
            dir_label = f"{sign_name}{axis_name}"

            # 多间隙尝试（以格距为单位，0~6格间距）
            for gap_units in range(0, 7):
                gap_mm = gap_units * grid_spacing

                if axis_dir == 0:  # X 轴偏移
                    if sign == +1:
                        ext_col_min = col_max + 1 + gap_units
                        ext_col_max = ext_col_min + n_cols1 - 1
                    else:
                        ext_col_max = col_min - 1 - gap_units
                        ext_col_min = ext_col_max - n_cols1 + 1
                    ext_row_min = row_min
                    ext_row_max = row_max
                else:  # Y 轴偏移
                    if sign == +1:
                        ext_row_min = row_max + 1 + gap_units
                        ext_row_max = ext_row_min + n_rows1 - 1
                    else:
                        ext_row_max = row_min - 1 - gap_units
                        ext_row_min = ext_row_max - n_rows1 + 1
                    ext_col_min = col_min
                    ext_col_max = col_max

                # 生成假定第二棋盘格的世界坐标网格
                ext_cols = np.arange(ext_col_min, ext_col_max + 1)
                ext_rows = np.arange(ext_row_min, ext_row_max + 1)
                if len(ext_cols) < 3 or len(ext_rows) < 3:
                    continue

                cc, rr = np.meshgrid(ext_cols, ext_rows)
                ext_world = np.column_stack([
                    cc.ravel() * grid_spacing,
                    rr.ravel() * grid_spacing
                ])

                # 正向投影到图像
                projected = _project_world_points(first_H, ext_world)
                if projected is None:
                    continue

                # 去掉投影超出图像边界的
                finite = np.isfinite(projected).all(axis=1)
                if np.sum(finite) < min_points:
                    continue

                projected_f = projected[finite]
                ext_world_f = ext_world[finite]

                # 匹配：对每个投影点，找最近的候选点
                # 匹配阈值：取投影网格的平均间距的 55%（够宽松）
                grid_spacing_img = _projected_grid_spacing(
                    projected_f.reshape(len(ext_rows), len(ext_cols), 2),
                    ext_cols, ext_rows
                ) if np.sum(finite) >= len(ext_cols) * len(ext_rows) else 25.0
                match_dist = max(8.0, min(35.0, grid_spacing_img * 0.55))

                dists, idxs = tree.query(projected_f, k=1)

                good = dists <= match_dist
                if np.sum(good) < min_points:
                    continue

                matched_img = candidate_pts[idxs[good]]
                matched_world = ext_world_f[good]

                # 去重：同一候选点被多个投影匹配时，保留距离最小的
                dedup = {}
                for i in range(len(matched_img)):
                    pid = int(idxs[good][i])
                    d = float(dists[good][i])
                    if pid not in dedup or d < dedup[pid][1]:
                        dedup[pid] = (i, d)
                idx_list = [v[0] for v in dedup.values()]
                matched_img = matched_img[idx_list]
                matched_world = matched_world[idx_list]

                if len(matched_img) < min_points:
                    continue

                # 世界坐标归零
                shift = np.min(matched_world, axis=0)
                matched_world_s = matched_world - shift

                # 质量评分
                snap = np.round(matched_world_s / grid_spacing).astype(int)
                n_c = len(np.unique(snap[:, 0]))
                n_r = len(np.unique(snap[:, 1]))
                if n_c < 2 or n_r < 2:
                    continue

                quality = len(matched_img) * (1.0 + 0.25 * min(n_c, n_r))

                if best_result is None or quality > best_result[0]:
                    H_cand = compute_homography(matched_world_s, matched_img)
                    if H_cand is not None:
                        best_result = (quality, matched_img,
                                       matched_world_s, H_cand, dir_label, match_dist)

    if best_result is None:
        print("  [传播] 未找到第二个棋盘格")
        return None

    quality, img_pts, world_pts, H, dir_name, mdist = best_result
    print(f"  [传播·{dir_name}] 推定 {len(img_pts)} 网格点, 匹配阈值={mdist:.1f}px")
    return img_pts, world_pts, H, set()


def fill_missing_saddles_by_proximity(img_pts, world_pts, all_saddle_pts, grid_spacing=15.0):
    """
    近邻吸附：对每个孤儿鞍点，在棋盘格上找k个最近点，尝试所有点对组合。
    若鞍点在任一点对连线上（垂距<12px，投影t在0.02~0.98），则用线性插值估算世界坐标并附加。

    返回: (extended_img, extended_world, n_added)
    """
    if len(img_pts) < 4 or len(all_saddle_pts) == 0:
        return img_pts, world_pts, 0

    # 估算网格间距
    diffs = np.linalg.norm(np.diff(img_pts[:min(10, len(img_pts))], axis=0), axis=1)
    img_spacing = float(np.median(diffs)) if len(diffs) > 0 else 25.0

    tree_board = KDTree(img_pts)
    tree_saddle = KDTree(all_saddle_pts)

    dist_board_to_saddle, _ = tree_saddle.query(img_pts, k=1)
    typical_match_dist = float(np.median(dist_board_to_saddle)) + 3.0

    # 孤儿鞍点：距棋盘格 > typical_match_dist 且 < 5倍间距（覆盖更大的棋盘格）
    k_near = min(5, len(img_pts))
    dist_to_board, idx_nearest = tree_board.query(all_saddle_pts, k=k_near)
    orphan_mask = (dist_to_board[:, 0] > typical_match_dist) & (dist_to_board[:, 0] < img_spacing * 5.0)
    orphan_idx = np.where(orphan_mask)[0]

    if len(orphan_idx) == 0:
        return img_pts, world_pts, 0

    added_pts = []
    added_world = []
    added_img_set = set()  # 用坐标元组去重
    occupied_world = {
        tuple(np.round(w / grid_spacing).astype(int)) for w in world_pts
    }

    for oi in orphan_idx:
        cand = all_saddle_pts[oi]
        found = False
        # 尝试k个最近点中所有点对组合
        for a in range(k_near):
            if found:
                break
            i1 = int(idx_nearest[oi, a])
            pt1 = img_pts[i1]
            w1 = world_pts[i1]
            for b in range(a + 1, k_near):
                i2 = int(idx_nearest[oi, b])
                pt2 = img_pts[i2]
                w2 = world_pts[i2]

                # Only fill a genuinely missing node between two points on
                # the same physical world row/column.  The previous code used
                # any nearby image-space pair, so diagonal pairs generated
                # fractional/duplicate world labels and pulled exterior
                # saddles into a board (hewei180--182).
                g1 = np.round(w1 / grid_spacing).astype(int)
                g2 = np.round(w2 / grid_spacing).astype(int)
                dg = np.abs(g2 - g1)
                same_grid_line = ((dg[0] == 0 and 2 <= dg[1] <= 4) or
                                  (dg[1] == 0 and 2 <= dg[0] <= 4))
                if not same_grid_line:
                    continue

                vec_12 = pt2 - pt1
                dist_12 = float(np.linalg.norm(vec_12))
                if dist_12 < 5.0:
                    continue

                t = float(np.dot(cand - pt1, vec_12) / (dist_12 ** 2))
                if t < 0.02 or t > 0.98:
                    continue

                proj = pt1 + t * vec_12
                perp_dist = float(np.linalg.norm(cand - proj))
                if perp_dist > 12.0:
                    continue

                # 通过：线性插值世界坐标
                w_new = w1 + t * (w2 - w1)
                w_snap = np.round(w_new / grid_spacing) * grid_spacing
                if np.linalg.norm(w_new - w_snap) > 2.5:
                    continue
                world_key = tuple(np.round(w_snap / grid_spacing).astype(int))
                if world_key in occupied_world:
                    continue
                key = (round(cand[0], 1), round(cand[1], 1))
                if key not in added_img_set:
                    added_img_set.add(key)
                    added_pts.append(cand)
                    added_world.append(w_snap)
                    occupied_world.add(world_key)
                found = True
                break

    if len(added_pts) == 0:
        return img_pts, world_pts, 0

    new_pts = np.array(added_pts)
    new_world = np.array(added_world)
    combined_img = np.vstack([img_pts, new_pts])
    combined_world = np.vstack([world_pts, new_world])
    combined_world = combined_world - np.min(combined_world, axis=0)

    return combined_img, combined_world, len(new_pts)


def process_single_image(img_path, divide_n=2, show=True, save_dir=None):
    global img_path_glob
    img_path_glob = img_path
    print(f"\n{'=' * 60}")
    print(f"处理图片: {os.path.basename(img_path)}")
    print(f"{'=' * 60}")
    img_gray = imread_gray(img_path)
    img_gray = gaussian_filter(img_gray, sigma=0.5)
    blur_var = measure_blur_level(img_gray)
    tau_nms = 0.015 if blur_var < 50 else 0.02
    tau_score = 0.01 if blur_var < 50 else 0.015
    print(f"  模糊度={blur_var:.1f}, NMS阈值={tau_nms}, 评分阈值={tau_score}")
    h, w = img_gray.shape
    nn = [h, w]
    mm = h
    jj3, ii3 = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing='xy')
    spij3 = (ii3.ravel(order='F') * h + jj3.ravel(order='F')).astype(int)
    jj7, ii7 = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8), indexing='xy')
    spij7 = (ii7.ravel(order='F') * h + jj7.ravel(order='F')).astype(int)
    Im0 = img_gray.ravel(order='F')
    t0 = time.time()
    angle_map, mag_map, img_du, img_dv = compute_gradients(img_gray)
    img_norm = normalize_image_for_corners(img_gray)
    radius_list = [4, 8, 12]
    corner_map = filter_image_with_templates_fft(img_norm, radius_list)
    corners_init = non_maximum_suppression_fast(corner_map, n=3, tau=tau_nms, margin=2)
    if len(corners_init) == 0:
        print("警告: 未检测到候选点，跳过")
        return True, img_path
    MAX_CANDIDATES = 3000
    if len(corners_init) > MAX_CANDIDATES:
        resp_vals = corner_map[corners_init[:, 1].astype(int), corners_init[:, 0].astype(int)]
        idx = np.argsort(resp_vals)[-MAX_CANDIDATES:]
        corners_init = corners_init[idx]
    refined_p, refined_v1, refined_v2 = refine_corners_vectorized(
        img_du, img_dv, angle_map, mag_map, corners_init, r=10, img_gray=img_norm)
    if len(refined_p) == 0:
        print("警告: 精化后无有效点，跳过")
        return True, img_path
    scores = score_corners_batch(img_norm, mag_map, refined_p, refined_v1, refined_v2, radius_list)
    keep = scores >= tau_score
    final_points = refined_p[keep]
    final_v1 = refined_v1[keep]
    final_v2 = refined_v2[keep]
    final_scores = scores[keep]
    print(f"  候选点数: NMS后={len(corners_init)}, 精化后={len(refined_p)}, 评分过滤后={len(final_points)}")
    if len(final_points) == 0:
        print("警告: 评分过滤后无点，跳过")
        return True, img_path
    dirs_array = np.stack([final_v1, final_v2], axis=1)
    raw_points = final_points
    raw_scores = final_scores
    final_points, dirs_array = merge_close_points(
        final_points, dirs_array, final_scores, dist_thresh=8.0, ratio=0.5
    )
    score_tree = KDTree(raw_points)
    _, score_idx = score_tree.query(final_points, k=1)
    final_scores = raw_scores[score_idx]
    final_v1 = dirs_array[:, 0]
    final_v2 = dirs_array[:, 1]
    final_v1, final_v2 = adjust_orientation_handedness(final_v1, final_v2)
    print(f"鞍点检测耗时: {time.time() - t0:.2f} 秒")
    corner = {'v1': final_v1, 'v2': final_v2}
    xcsp = np.round(final_points[:, 0]).astype(int) * h + np.round(final_points[:, 1]).astype(int)
    # ========== 4_29版本棋盘格检测：get_little_four_P_py + grow_chessboard_region ==========
    # 4_29方法核心：单应投影生长，无BFS方向约束，天然处理透视畸变和角落鞍点
    points = final_points.copy()
    dirs_4_29 = np.stack([final_v1, final_v2], axis=1)  # (N,2,2) 格式
    found_chessboards_raw = []
    sorted_idx = np.argsort(final_scores)[::-1]

    for start_idx in sorted_idx[:min(500, len(points))]:
        try:
            quad_pts, quad_idx = get_little_four_P_py(points, dirs_4_29, start_idx, img_gray.shape)
        except Exception:
            continue
        if quad_idx is None:
            continue
        try:
            img_pts, world_pts, H, used_idx = grow_chessboard_region(
                points, dirs_4_29, quad_idx, max_iter=120, dist_thresh=12.0)
        except Exception:
            continue
        if len(img_pts) < 6:
            continue
        img_pts, world_pts = _unique_world_points(img_pts, world_pts)
        if len(img_pts) < 6:
            continue
        # 归一化世界坐标（4_29方式：减原点，必要时交换XY轴）
        world_pts = world_pts - np.min(world_pts, axis=0)
        max_xy = np.max(world_pts, axis=0)
        if max_xy[0] > max_xy[1]:
            world_pts = world_pts[:, [1, 0]]
            world_pts[:, 0] = 15.0 - world_pts[:, 0]
        H = compute_homography(world_pts, img_pts)
        if H is None:
            continue
        # 简单去重（4_29方式：>50%共享鞍点即判为重复）
        used_idx_set = set(used_idx)
        is_new = True
        for exist_img, exist_world, exist_H, exist_used in found_chessboards_raw:
            overlap = len(used_idx_set & exist_used)
            if overlap > 0.5 * len(used_idx_set) or overlap > 0.5 * len(exist_used):
                is_new = False
                break
        if is_new:
            found_chessboards_raw.append((img_pts, world_pts, H, used_idx_set))
        if len(found_chessboards_raw) >= 12:
            break

    if not found_chessboards_raw:
        print("未找到有效棋盘格")
        return False, img_path

    # 主候选也必须经过跳格检查。BFS 有时会隔行或隔列生长，若在这里
    # 直接进入评分，后续位于间隔中的真实鞍点会被投影回已经占用的世界
    # 网格，既无法补入棋盘，还会造成红线把相邻物理列误当成同一列。
    # prepare 只在凸包内有足够未归属候选且它们稳定落在奇数网格时才
    # 加密坐标轴；完整密集棋盘不会触发该修正。
    dirs = np.column_stack([final_v1, final_v2])  # BFS格式（供策略B/C使用）
    scored_boards = []
    for img_pts, world_pts, H, used_idx in found_chessboards_raw:
        img_pts, world_pts, H = prepare_chessboard_candidate(
            img_pts, world_pts, final_points, H)
        if len(img_pts) < 4:
            continue
        world_pts = world_pts - np.min(world_pts, axis=0)
        # Score overlapping seed models after a one-cell, real-candidate-only
        # completion.  A sparse but correctly scaled 7x9 model can start with
        # fewer seed points than a compressed 4x11 model; raw seed count would
        # otherwise select the compressed topology and lose whole columns
        # (hewei185).  complete_chessboard_from_candidates matches only actual
        # saddle candidates, so no synthetic display point is introduced here.
        preview_img, preview_world, preview_H = complete_chessboard_from_candidates(
            img_pts, world_pts, final_points, expand_margin=1)
        preview_img, preview_world = _filter_points_inside_polygon(
            preview_img, preview_world, img_pts, expand_ratio=0.06)
        if len(preview_img) > len(img_pts):
            preview_p90, _, preview_bad = chessboard_line_residual_summary(
                preview_img, preview_world)
            if preview_bad == 0 and preview_p90 <= 3.0:
                img_pts, world_pts = preview_img, preview_world
                H_candidate = compute_homography(world_pts, img_pts)
                if H_candidate is not None:
                    H = H_candidate
        # 网格验证放宽（4_29不验证，这里用宽松阈值保留所有有效板）
        valid, med_dev = validate_chessboard_grid_alignment(world_pts, max_deviation=5.0)
        if not valid and len(img_pts) < 6:
            pass  # 小板宽容
        quality = calculate_chessboard_quality(img_pts, world_pts)
        p90_line, worst_line, bad_groups = chessboard_line_residual_summary(img_pts, world_pts)
        print(
            f"  候选棋盘: 点数 {len(img_pts)}, 线残差p90 {p90_line:.2f}, "
            f"最大残差 {worst_line:.2f}, 异常组 {bad_groups}, 质量 {quality:.2f}")
        scored_boards.append((img_pts, world_pts, H, used_idx, quality))
    # 按质量降序，取不重叠的棋盘格（数量不限，后续由MAX_BOARDS控制总量）
    # ⚠ 重要：用图像空间重叠检查，而非世界坐标（世界坐标已归零，不同棋盘格可能
    #    巧合重叠）
    MAX_BOARDS = 6  # 提升：从3→6，识别更多棋盘格
    PROPAGATION_MIN_QUALITY = 5.0  # 传播新增棋盘格的最低质量阈值
    # Among overlapping, geometrically valid models, reward real completed
    # support in addition to homography quality.  Otherwise a compressed
    # 5-column model can outrank a 7-column model containing 25 more genuine
    # saddles solely because the latter's initial world labels have a larger
    # reprojection penalty (hewei207).
    scored_boards.sort(
        key=lambda x: x[4] + 0.50 * len(x[0]), reverse=True)
    final_two = []
    used_img_ranges = []
    for img_pts, world_pts, H, used_idx, qual in scored_boards:
        # 重叠检查 — 基于图像坐标
        ix_min, ix_max = np.min(img_pts[:, 0]), np.max(img_pts[:, 0])
        iy_min, iy_max = np.min(img_pts[:, 1]), np.max(img_pts[:, 1])
        img_area = max(1.0, (ix_max - ix_min) * (iy_max - iy_min))
        cx = float(np.mean(img_pts[:, 0]))
        cy = float(np.mean(img_pts[:, 1]))

        overlap = False
        for (ox_min, ox_max, oy_min, oy_max) in used_img_ranges:
            inter_x = max(0.0, min(ix_max, ox_max) - max(ix_min, ox_min))
            inter_y = max(0.0, min(iy_max, oy_max) - max(iy_min, oy_min))
            if inter_x * inter_y > 0.15 * img_area:
                overlap = True
                break

        # 额外保护：用图像点集 KDTree 精细判断重复（同一棋盘格不同算法产出）
        if not overlap and len(final_two) >= 1:
            for (exist_img, _, _, _, _) in final_two:
                tree_exist = KDTree(exist_img)
                dists, _ = tree_exist.query(img_pts, k=1)
                if np.mean(dists < 12.0) > 0.65:
                    overlap = True
                    break

        # 质量过滤：拒绝低质量假棋盘格
        if qual < 0.5:
            print(f"  候选: {len(img_pts)}点 q={qual:.2f} "
                  f"img中心=({cx:.0f},{cy:.0f}) "
                  f"img_X[{ix_min:.0f},{ix_max:.0f}] img_Y[{iy_min:.0f},{iy_max:.0f}] "
                  f"→ 质量过低跳过")
            continue

        status = "已选" if (not overlap) else "重叠跳过"
        print(f"  候选: {len(img_pts)}点 q={qual:.2f} "
              f"img中心=({cx:.0f},{cy:.0f}) "
              f"img_X[{ix_min:.0f},{ix_max:.0f}] img_Y[{iy_min:.0f},{iy_max:.0f}] "
              f"→ {status}")
        if not overlap:
            final_two.append((img_pts, world_pts, H, used_idx, qual))
            used_img_ranges.append((ix_min, ix_max, iy_min, iy_max))

    print(f"  不重叠棋盘格: {len(final_two)} 个")

    # A foot can hide one side/one end of a narrow board so the normal
    # four-corner seed never forms.  Recover repeated two-column row pairs
    # from unassigned genuine saddle responses before propagation.
    if len(final_two) < MAX_BOARDS:
        strip_boards = detect_two_column_strip_fallback(points, final_two, min_pairs=3)
        for s_img, s_world, s_H, s_used in strip_boards:
            s_quality = calculate_chessboard_quality(s_img, s_world)
            final_two.append((s_img, s_world, s_H, s_used, s_quality))
            print(f"  [两列回退] 已加入棋盘候选，质量={s_quality:.2f}")
            if len(final_two) >= MAX_BOARDS:
                break

    # More severe occlusion can leave only one complete row pair followed by
    # a stable run in one column.  Keep this separate from the generic quality
    # threshold: its evidence is regular one-dimensional spacing plus the
    # real row partner, rather than a four-corner homography seed.
    if len(final_two) < MAX_BOARDS:
        one_sided = detect_one_sided_strip_fallback(
            points, final_two, min_chain=5, point_scores=final_scores)
        for s_img, s_world, s_H, s_used in one_sided:
            s_quality = max(calculate_chessboard_quality(s_img, s_world),
                            0.90 * len(s_img))
            final_two.append((s_img, s_world, s_H, s_used, s_quality))
            print(f"  [单侧两列回退] 已加入棋盘候选，质量={s_quality:.2f}")
            if len(final_two) >= MAX_BOARDS:
                break

    # If no genuine same-row partner survives, a complete butterfly plus a
    # regular vertical saddle run can still prove the visible part of the
    # narrow physical strip.  Keep it as a one-column display topology so no
    # hidden/interpolated saddle is drawn or connected.
    if len(final_two) < MAX_BOARDS:
        x_guided = detect_x_guided_single_column_fallback(
            points, final_two, img_gray, point_scores=final_scores)
        for s_img, s_world, s_H, s_used in x_guided:
            s_quality = 0.85 * len(s_img)
            final_two.append((s_img, s_world, s_H, s_used, s_quality))
            print(f"  [X引导单列回退] 已加入棋盘候选，质量={s_quality:.2f}")
            if len(final_two) >= MAX_BOARDS:
                break

    # ========== 循环传播：找最多 MAX_BOARDS 个棋盘格 ==========
    # 设计为可扩展的循环：每轮用一个已识别的棋盘格传播新棋盘
    # 用所有已识别棋盘格的合并凸包做排除
    if len(final_two) >= 1 and len(final_two) < MAX_BOARDS:
        print(f"  当前已识别 {len(final_two)}/{MAX_BOARDS}，尝试传播找更多...")

        # 多轮传播
        propagation_attempts = 0
        max_propagation_attempts = MAX_BOARDS + 1  # 防止死循环

        while len(final_two) < MAX_BOARDS and propagation_attempts < max_propagation_attempts:
            propagation_attempts += 1
            found_this_round = False

            # ---- 计算所有已识别棋盘格的合并凸包掩码 ----
            all_excluded_mask = np.zeros(len(final_points), dtype=bool)
            for (exist_img, _, _, _, _) in final_two:
                poly, _ = get_chessboard_polygon(exist_img)
                if poly is not None:
                    all_excluded_mask |= poly.contains_points(final_points)

            outside_mask = ~all_excluded_mask
            outside_points = final_points[outside_mask]

            if len(outside_points) < 6:
                print(f"  凸包排除后剩余 {len(outside_points)} 点，跳过")
                break

            # ---- 策略 A：单应传播 ----
            # 尝试用每个已识别棋盘格传播
            best_propagated = None
            best_extension = None
            for src_idx, (src_img, src_world, src_H, _, _) in enumerate(final_two):
                # Always propagate with a homography fitted to the current
                # accepted topology.  The stored matrix can predate lattice
                # completion or origin normalization and is then unsuitable
                # for deciding whether a result is the source's next column.
                current_src_H = compute_homography(src_world, src_img)
                if current_src_H is None:
                    current_src_H = src_H
                propagated = propagate_chessboard_by_homography(
                    src_img, src_world, current_src_H, final_points,
                    grid_spacing=15.0, min_points=6
                )
                if propagated is not None:
                    p_img, p_world, p_H, _ = propagated
                    # A propagated result immediately outside exactly one
                    # source boundary is an omitted row/column of that same
                    # physical board, not a second board.  Map it back with
                    # the source homography and retain only clean one-cell
                    # extensions; this also rejects isolated far-grid points.
                    try:
                        src_H_inv = np.linalg.inv(current_src_H)
                    except (TypeError, np.linalg.LinAlgError):
                        src_H_inv = None
                    if src_H_inv is not None and len(p_img) > 0:
                        p_h = np.column_stack([p_img, np.ones(len(p_img))])
                        src_w_h = (src_H_inv @ p_h.T).T
                        finite_w = np.abs(src_w_h[:, 2]) > 1e-9
                        src_units = np.full((len(p_img), 2), np.nan, dtype=float)
                        src_units[finite_w] = (
                            src_w_h[finite_w, :2] /
                            src_w_h[finite_w, 2, np.newaxis] / 15.0)
                        snapped_units = np.zeros((len(p_img), 2), dtype=int)
                        snapped_units[finite_w] = np.round(
                            src_units[finite_w]).astype(int)
                        unit_error = np.full(len(p_img), np.inf, dtype=float)
                        unit_error[finite_w] = np.linalg.norm(
                            src_units[finite_w] - snapped_units[finite_w], axis=1)
                        source_grid = np.round(src_world / 15.0).astype(int)
                        c_min, r_min = np.min(source_grid, axis=0)
                        c_max, r_max = np.max(source_grid, axis=0)
                        # Accept both missing cells inside the current extent
                        # and a genuine one-cell border extension.  Previously
                        # only the latter was merged, so a propagated set that
                        # filled nine holes in the last existing row became a
                        # second board and then spawned a third overlapping
                        # board (hewei190).
                        in_local_extent = (
                            (snapped_units[:, 0] >= c_min - 1) &
                            (snapped_units[:, 0] <= c_max + 1) &
                            (snapped_units[:, 1] >= r_min - 1) &
                            (snapped_units[:, 1] <= r_max + 1))
                        # A strongly tilted/compressed boundary can accumulate
                        # about 0.30 grid-unit extrapolation error even though
                        # an entire run maps to one adjacent source column
                        # (hewei207).  The local one-cell extent, unique-cell
                        # assignment and 60% coherent-support gate keep 0.32
                        # below the half-cell ambiguity limit while allowing
                        # that genuine boundary column to merge.
                        candidate_mask = finite_w & (unit_error < 0.32) & in_local_extent
                        occupied_cells = {tuple(cell) for cell in source_grid}
                        selected_by_cell = {}
                        for point_idx in np.where(candidate_mask)[0]:
                            key = tuple(snapped_units[int(point_idx)])
                            if key in occupied_cells:
                                continue
                            if (key not in selected_by_cell or
                                    unit_error[point_idx] < unit_error[selected_by_cell[key]]):
                                selected_by_cell[key] = int(point_idx)
                        extension_mask = np.zeros(len(p_img), dtype=bool)
                        if selected_by_cell:
                            extension_mask[list(selected_by_cell.values())] = True
                        extension_count = int(np.sum(extension_mask))
                        if extension_count >= max(4, int(np.ceil(0.60 * len(p_img)))):
                            extension_score = (
                                extension_count -
                                float(np.mean(unit_error[extension_mask])))
                            if (best_extension is None or
                                    extension_score > best_extension[0]):
                                best_extension = (
                                    extension_score, src_idx,
                                    p_img[extension_mask],
                                    snapped_units[extension_mask].astype(float) * 15.0)
                            continue
                    # The same extrapolated strip can be returned again in a
                    # later propagation round.  Do not append it as another
                    # board: repeated boards duplicate display nodes and can
                    # make one physical edge appear several times.
                    duplicate_propagation = False
                    for exist_img, _, _, _, _ in final_two:
                        if len(exist_img) == 0:
                            continue
                        dist_existing, _ = KDTree(exist_img).query(p_img, k=1)
                        if np.mean(dist_existing < 8.0) > 0.65:
                            duplicate_propagation = True
                            break
                    if duplicate_propagation:
                        continue
                    # 检验：传播结果必须在凸包外
                    poly_src, _ = get_chessboard_polygon(src_img)
                    if poly_src is not None:
                        # 至少 50% 的点不在 src 棋盘格内
                        outside_ratio = np.mean(~poly_src.contains_points(p_img))
                        if outside_ratio < 0.5:
                            continue
                    # 质量优先
                    p_world_s = p_world - np.min(p_world, axis=0)
                    valid, med_dev = validate_chessboard_grid_alignment(
                        p_world_s, max_deviation=3.5)
                    if not valid or len(p_img) < 6:
                        continue
                    qual = calculate_chessboard_quality(p_img, p_world_s)
                    if best_propagated is None or qual > best_propagated[0]:
                        best_propagated = (qual, p_img, p_world_s, p_H, src_idx)

            if best_extension is not None:
                _, src_idx, ext_img, ext_world = best_extension
                src_img, src_world, src_H, src_used, src_qual = final_two[src_idx]
                merged_img = np.vstack([src_img, ext_img])
                merged_world = np.vstack([src_world, ext_world])
                merged_img, merged_world = _unique_world_points(
                    merged_img, merged_world)
                merged_H = compute_homography(merged_world, merged_img)
                merged_qual = max(
                    src_qual, calculate_chessboard_quality(merged_img, merged_world))
                final_two[src_idx] = (
                    merged_img, merged_world,
                    merged_H if merged_H is not None else src_H,
                    src_used, merged_qual)
                found_this_round = True
                print(f"  [OK] 相邻边界并回棋盘 #{src_idx + 1}: "
                      f"+{len(ext_img)}点 -> {len(merged_img)}点")
                continue

            if best_propagated is not None:
                qual, p_img, p_world, p_H, src_idx = best_propagated
                if qual < PROPAGATION_MIN_QUALITY:
                    print(f"  [SKIP] 策略A最佳传播棋盘格质量过低 (q={qual:.2f} < {PROPAGATION_MIN_QUALITY:.1f})，跳过")
                else:
                    final_two.append((p_img, p_world, p_H, set(), qual))
                    found_this_round = True
                    print(f"  [OK] 策略A单应传播: 棋盘格 #{len(final_two)} "
                          f"(源自#{src_idx+1}, {len(p_img)}点, 质量{qual:.1f})")
                    continue


            # ---- 策略 B：凸包排除后重搜索 ----
            outside_v1 = final_v1[outside_mask]
            outside_v2 = final_v2[outside_mask]
            outside_scores_arr = final_scores[outside_mask]
            outside_dirs = np.column_stack([outside_v1, outside_v2])

            second_boards = find_chessboard_candidates(
                outside_points, outside_dirs, outside_scores_arr,
                img_gray.shape, max_boards=1
            )
            for b_img, b_world, b_H, b_used in second_boards:
                b_img, b_world, b_H = prepare_chessboard_candidate(
                    b_img, b_world, outside_points, b_H)
                b_world = b_world - np.min(b_world, axis=0)
                valid2, _ = validate_chessboard_grid_alignment(b_world)
                if valid2 and len(b_img) >= 6:
                    qual2 = calculate_chessboard_quality(b_img, b_world)
                    if qual2 < PROPAGATION_MIN_QUALITY:
                        print(f"  [SKIP] 策略B棋盘格质量过低 (q={qual2:.2f} < {PROPAGATION_MIN_QUALITY:.1f})，跳过")
                        continue
                    final_two.append((b_img, b_world, b_H, set(), qual2))
                    found_this_round = True
                    print(f"  [OK] 策略B凸包重搜索: 棋盘格 #{len(final_two)} "
                          f"({len(b_img)}点, 质量{qual2:.1f})")
                    break

            if found_this_round:
                continue

            # ---- 策略 C：降低 tau_score 阈值 ----
            lower_tau = tau_score * 0.55
            keep_lower = scores >= lower_tau
            if np.sum(keep_lower) > np.sum(keep):
                ext_points_all = refined_p[keep_lower]
                ext_v1_all = refined_v1[keep_lower]
                ext_v2_all = refined_v2[keep_lower]
                ext_scores_all = scores[keep_lower]
                ext_dirs_all = np.column_stack([ext_v1_all, ext_v2_all])

                # 合并去重
                merged_p, merged_dirs = merge_close_points(
                    ext_points_all, ext_dirs_all, ext_scores_all,
                    dist_thresh=8.0, ratio=0.5)
                tree_raw = KDTree(refined_p)
                _, sidx = tree_raw.query(merged_p, k=1)
                merged_scores = scores[sidx]
                merged_v1 = merged_dirs[:, :2]
                merged_v2 = merged_dirs[:, 2:4]

                # 排除所有已识别棋盘格凸包
                poly_all = []
                for (exist_img, _, _, _, _) in final_two:
                    poly, _ = get_chessboard_polygon(exist_img)
                    if poly is not None:
                        poly_all.append(poly)

                if poly_all:
                    outside_mask_m = np.ones(len(merged_p), dtype=bool)
                    for poly in poly_all:
                        outside_mask_m &= ~poly.contains_points(merged_p)
                else:
                    outside_mask_m = np.ones(len(merged_p), dtype=bool)

                if np.sum(outside_mask_m) >= 6:
                    o_points = merged_p[outside_mask_m]
                    o_v1 = merged_v1[outside_mask_m]
                    o_v2 = merged_v2[outside_mask_m]
                    o_scores = merged_scores[outside_mask_m]
                    o_dirs = np.column_stack([o_v1, o_v2])

                    extra_boards = find_chessboard_candidates(
                        o_points, o_dirs, o_scores,
                        img_gray.shape, max_boards=1
                    )
                    for b_img, b_world, b_H, b_used in extra_boards:
                        b_img, b_world, b_H = prepare_chessboard_candidate(
                            b_img, b_world, o_points, b_H)
                        b_world = b_world - np.min(b_world, axis=0)
                        valid2, _ = validate_chessboard_grid_alignment(
                            b_world, max_deviation=3.0)
                        if valid2 and len(b_img) >= 6:
                            qual2 = calculate_chessboard_quality(b_img, b_world)
                            if qual2 < PROPAGATION_MIN_QUALITY:
                                print(f"  [SKIP] 策略C棋盘格质量过低 (q={qual2:.2f} < {PROPAGATION_MIN_QUALITY:.1f})，跳过")
                                continue
                            final_two.append((b_img, b_world, b_H, set(), qual2))
                            found_this_round = True
                            print(f"  [OK] 策略C降阈值重检: 棋盘格 #{len(final_two)} "
                                  f"({len(b_img)}点, 质量{qual2:.1f})")
                            break

            if not found_this_round:
                print(f"  [X] 本轮所有策略失败，停止传播")
                break

        print(f"  传播完成: 共识别 {len(final_two)}/{MAX_BOARDS} 个棋盘格")
    # ================================================================

    cleaned_final_two = []
    for img_pts, world_pts, H, used_idx, qual in final_two:
        orig_img = img_pts.copy()
        stable_H = H if _homography_has_grid_scale(H, world_pts) else None

        # 外扩 margin=4：单应投影外扩4格以捕获生长遗漏的边界/角落鞍点
        clean_img, clean_world, clean_H = complete_chessboard_from_candidates(
            img_pts, world_pts, final_points, expand_margin=4)
        if not _homography_has_grid_scale(clean_H, clean_world):
            clean_H = stable_H
        completed_img = clean_img.copy()
        completed_world = clean_world.copy()
        # 仅小幅外扩：足够容纳亚像素误差，同时避免吸入相邻棋盘。
        clean_img, clean_world = _filter_points_inside_polygon(
            clean_img, clean_world, orig_img, expand_ratio=0.06)
        clean_img, clean_world, restored_end = _restore_complete_narrow_end_rows(
            clean_img, clean_world, completed_img, completed_world, world_pts)
        if restored_end > 0:
            candidate_H = compute_homography(clean_world, clean_img)
            if _homography_has_grid_scale(candidate_H, clean_world):
                clean_H = candidate_H
            print(f"  两列棋盘端行恢复: +{restored_end}个真实鞍点 -> {len(clean_img)}点")

        before_fill = len(clean_img)
        snapped = np.round(clean_world / 15.0).astype(int)
        grid_cells = len(np.unique(snapped[:, 0])) * len(np.unique(snapped[:, 1]))
        fill_ratio = len(clean_img) / max(1, grid_cells)
        min_n = 0 if fill_ratio > 0.70 else 2
        clean_img, clean_world = interpolate_missing_grid_points(clean_img, clean_world, min_neighbors=min_n)

        if len(clean_img) != before_fill:
            print(f"  网格插值填补: {before_fill} -> {len(clean_img)} 个交叉点")
            candidate_H = compute_homography(clean_world, clean_img)
            if _homography_has_grid_scale(candidate_H, clean_world):
                clean_H = candidate_H

        # ---- 鞍点吸附：将棋盘格附近的孤立鞍点附加进来 ----
        if clean_H is not None and len(clean_img) >= 4:
            tree_board = KDTree(clean_img)
            dist_to_board, _ = tree_board.query(final_points, k=1)
            # 只在一个典型网格步长内吸附，避免跨到邻近棋盘。
            _, nn_idx = tree_board.query(clean_img, k=2)
            local_spacing = float(np.median(np.linalg.norm(
                clean_img - clean_img[nn_idx[:, 1]], axis=1)))
            attach_radius = float(np.clip(0.80 * local_spacing, 8.0, 32.0))
            orphan_mask = (dist_to_board > 3.0) & (dist_to_board < attach_radius)
            orphan_pts = final_points[orphan_mask]
            if len(orphan_pts) > 0:
                # 用单应投影验证：将孤立点投影到世界坐标，检查是否接近网格位置
                ones = np.ones((len(orphan_pts), 1))
                # 需要逆单应（image→world），用 H 的逆
                try:
                    H_inv = np.linalg.inv(clean_H)
                except np.linalg.LinAlgError:
                    H_inv = None
                if H_inv is not None:
                    img_h = np.hstack([orphan_pts, ones])
                    world_h = (H_inv @ img_h.T).T
                    world_proj = world_h[:, :2] / (world_h[:, 2, np.newaxis] + 1e-12)
                    # 检查投影是否接近15mm网格
                    snapped_proj = np.round(world_proj / 15.0) * 15.0
                    grid_dev = np.linalg.norm(world_proj - snapped_proj, axis=1)
                    # 网格偏差<3mm且投影在合理范围内
                    # 还必须落在当前棋盘已知世界范围的一格邻域内。
                    w_min = np.min(clean_world, axis=0) - 15.0
                    w_max = np.max(clean_world, axis=0) + 15.0
                    in_board_extent = np.all((world_proj >= w_min) & (world_proj <= w_max), axis=1)
                    valid_attach = (grid_dev < 3.0) & in_board_extent
                    if np.sum(valid_attach) > 0:
                        new_pts = orphan_pts[valid_attach]
                        new_world = snapped_proj[valid_attach]
                        # 避免重复：检查新点是否已在棋盘格中
                        tree_new = KDTree(new_pts)
                        dist_exist, _ = tree_board.query(new_pts, k=1)
                        not_dup = dist_exist > 3.0
                        if np.sum(not_dup) > 0:
                            clean_img = np.vstack([clean_img, new_pts[not_dup]])
                            clean_world = np.vstack([clean_world, new_world[not_dup]])
                            clean_world = clean_world - np.min(clean_world, axis=0)
                            candidate_H = compute_homography(clean_world, clean_img)
                            if _homography_has_grid_scale(candidate_H, clean_world):
                                clean_H = candidate_H
                            print(f"  鞍点吸附: +{np.sum(not_dup)}个孤立鞍点 -> {len(clean_img)}点")

        # ---- 近邻吸附：将棋盘格附近的孤立鞍点纳入 ----
        clean_img, clean_world, n_filled = fill_missing_saddles_by_proximity(
            clean_img, clean_world, final_points)
        if n_filled > 0:
            candidate_H = compute_homography(clean_world, clean_img)
            if _homography_has_grid_scale(candidate_H, clean_world):
                clean_H = candidate_H
            print(f"  近邻吸附: +{n_filled}个鞍点 -> {len(clean_img)}点")

        cleaned_final_two.append((clean_img, clean_world, clean_H if clean_H is not None else H, used_idx, qual))

    # A propagation result can be nothing more than the broad board's omitted
    # last column.  Merge such cleaned, geometrically proven extensions before
    # X detection and rendering, otherwise the same physical column is drawn
    # as a separate board and loses its red/blue neighbours.
    final_two = _merge_cleaned_broad_board_extensions(cleaned_final_two)
    final_two = _rebuild_broad_board_from_complete_image_rows(
        final_two, final_points)
    # 提取各棋盘格数据（支持任意数量）
    board_results = []  # [(xy, uv, jilu), ...]
    for (img_b, world_b, H_b, _, _) in final_two:
        # 强制归一化为参考代码格式：2 列竖直、x∈[0,15]、y 向下增长、按行排序
        world_n = world_b - np.min(world_b, axis=0)
        swapped_axes = False
        max_r = np.max(world_n, axis=0)
        if max_r[0] > max_r[1]:
            world_n = world_n[:, [1, 0]]
            world_n[:, 0] = 15.0 - world_n[:, 0]
            swapped_axes = True
        order = np.lexsort((world_n[:, 0], world_n[:, 1]))
        world_n = world_n[order]
        img_n = img_b[order]
        H_n = compute_homography(world_n, img_n)
        if (not _homography_has_grid_scale(H_n, world_n) and
                not swapped_axes and _homography_has_grid_scale(H_b, world_n)):
            H_n = H_b

        xy_b = np.column_stack([world_n, np.ones(len(world_n))])
        uv_b = np.column_stack([img_n, np.ones(len(img_n))])
        snapped_n = np.round(world_n / 15.0).astype(int)
        n_grid_columns = len(np.unique(snapped_n[:, 0]))
        if n_grid_columns == 2:
            xy_b, uv_b, raw_marks = get_mark_cord(
                final_points, corner, xy_b, uv_b, H_n,
                nn, spij3, spij7, [], xcsp, Im0, mm, img_gray)
            # First establish one complete two-lobe centre.  Filtering a raw
            # template coordinate before centring caused real marks near one
            # lobe to be rejected by a different stage's distance test.
            normal_marks = _refine_x_marks_multiscale(
                raw_marks, img_n, img_gray, min_center_score=0.50)
            # Large/small butterfly centres may not survive the saddle
            # templates, so search only along the two physical strip ends.
            fallback_marks = _detect_endpoint_butterfly_fallback(
                img_n, world_n, H_n, img_gray, normal_marks)
            fallback_marks = _refine_x_marks_multiscale(
                fallback_marks, img_n, img_gray, min_center_score=0.50)

            # One common post-condition for both sources: centre is outside
            # the saddle grid, on the board substrate, and has butterfly
            # contrast.  This replaces conflicting pre/post filter chains.
            combined = []
            dedup_dist = 0.45 * _estimate_grid_spacing(img_n)
            for rec in list(normal_marks) + list(fallback_marks):
                p = np.asarray(rec[1:3], dtype=float)
                if combined and min(np.linalg.norm(
                        p - np.asarray(old[1:3], dtype=float))
                        for old in combined) < dedup_dist:
                    continue
                combined.append(rec)
            jilu_b = _filter_false_positive_x_corners(
                combined, img_n, img_gray)
        else:
            jilu_b = []
            print(f"  [X跳过] 当前棋盘为 {n_grid_columns} 列，仅两列棋盘允许检测X角点")
        board_results.append((xy_b, uv_b, jilu_b))


    # A nearly fully occluded narrow strip may not have enough row pairs to
    # enter board_results.  Recover only butterflies that are independently
    # supported by a repeated vertical saddle strip below them.
    board_marks = [rec for (_, _, marks) in board_results for rec in marks]
    sparse_strip_marks = _detect_butterfly_above_sparse_strip(
        final_points, img_gray, final_two, board_marks)
    isolated_marks = _detect_isolated_complete_butterflies(
        final_points, img_gray, final_two,
        list(board_marks) + list(sparse_strip_marks))

    # All detector branches now share one physical numbering convention.
    board_results = [
        (xy_b, uv_b, _normalize_physical_x_corner_types(marks, img_gray.shape))
        for xy_b, uv_b, marks in board_results
    ]
    sparse_strip_marks = _normalize_physical_x_corner_types(
        sparse_strip_marks, img_gray.shape)
    isolated_marks = _normalize_physical_x_corner_types(
        isolated_marks, img_gray.shape)

    # 收集所有棋盘格实际检测到的X角点（不仅限于中央棋盘格，用于可视化）
    all_detected = list(sparse_strip_marks) + list(isolated_marks)
    for (_, _, jilu_b) in board_results:
        all_detected.extend([rec for rec in jilu_b])
    if len(all_detected) > 0:
        print(f"  所有棋盘格共检测到 {len(all_detected)} 个X角点")

    # 保持兼容旧变量名：取前 2 个
    xy1_left = uv1_left = None
    xy1_right = uv1_right = None
    jilu_left = []
    jilu_right = []
    if len(board_results) >= 1:
        xy1_left, uv1_left, jilu_left = board_results[0]
    if len(board_results) >= 2:
        xy1_right, uv1_right, jilu_right = board_results[1]

    def _check_handedness(pts, tol=1e-6):
        """
        从点集中选择一个非退化三角形，返回叉积符号。
        若所有点共线或点数不足，返回 0（无效）。
        """
        if pts is None or len(pts) < 3:
            return 0
        p0 = pts[0, :2]
        # 寻找第二个点，使向量长度 > tol
        for i in range(1, len(pts)):
            v1 = pts[i, :2] - p0
            if np.linalg.norm(v1) > tol:
                break
        else:
            return 0
        # 寻找第三个点，使与 p0, p1 形成非零面积
        for j in range(i + 1, len(pts)):
            v2 = pts[j, :2] - p0
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) > tol:
                return np.sign(cross)
        return 0

    # ========================================
    # ### 修改部分 1：移除旋向不一致导致的程序中止逻辑 ###
    # ========================================
    def check_handedness_ok(xy, uv, board_name):
        if xy is None or uv is None:
            return True
        s_xy = _check_handedness(xy)
        s_uv = _check_handedness(uv)
        if s_xy == 0 or s_uv == 0 or s_xy != s_uv:
            print(f"  警告: {board_name} 旋向异常 (xy {s_xy}, uv {s_uv})，已记录但不中止程序")
            return False
        return True

    # 仅检查并提示，不影响程序继续
    left_ok = check_handedness_ok(xy1_left, uv1_left, "左棋盘格")
    right_ok = check_handedness_ok(xy1_right, uv1_right, "右棋盘格")

    # 始终返回 True，让程序继续
    is_all_consistent = True
    # ========================================
    # ### 修改部分 1 结束 ###
    # ========================================

    # ===================== X角点：找出中央棋盘格（最靠近图像中心的棋盘格） =====================
    center_board_idx = -1
    center_jilu = []       # 全部角点（检测+推断），用于内部计算
    center_detected = []   # 仅检测到的角点（可视化用）
    center_inferred = []   # 仅推断的角点（不显示）

    # 以图像中心为基准，选择最靠近中心的棋盘格作为中央棋盘格
    img_center = np.array([img_gray.shape[1] / 2.0, img_gray.shape[0] / 2.0])
    board_centroids = []
    for b_idx, (im, wo, H_b, _, qual_b) in enumerate(final_two):
        if H_b is None or len(wo) < 4 or len(im) < 4:
            continue
        centroid = np.mean(im[:, :2], axis=0)
        dist = float(np.linalg.norm(centroid - img_center))
        board_centroids.append((dist, b_idx, im, wo, H_b, qual_b))

    if board_centroids:
        board_centroids.sort(key=lambda x: x[0])
        _, center_board_idx, center_im, center_wo, center_H, center_qual = board_centroids[0]

        # 从所有棋盘格中推测最完整的网格范围（用于修正遮挡导致的范围收缩）
        ref_col_max, ref_row_max = 0, 0
        for (_im, _wo, _Hb, _, _) in final_two:
            if _Hb is None or len(_wo) < 4 or len(_im) < 4:
                continue
            snapped_r = np.round(_wo / 15.0).astype(int)
            rcM = int(np.max(snapped_r[:, 0]))
            rrM = int(np.max(snapped_r[:, 1]))
            ref_col_max = max(ref_col_max, rcM)
            ref_row_max = max(ref_row_max, rrM)
        # 用最大范围修正中央棋盘格（但不能小于中央棋盘的检测范围）
        center_snapped = np.round(center_wo / 15.0).astype(int)
        full_col_max = max(int(np.max(center_snapped[:, 0])), ref_col_max)
        full_row_max = max(int(np.max(center_snapped[:, 1])), ref_row_max)

        # 在中央棋盘格上检测X角点（仅使用参考代码 get_mark_cord 的结果，
        # 它基于蝴蝶形黑白半圆标记的模板匹配，落在实际标记中心；
        # 没有蝴蝶形标记的图片不应标出X角点，因此不进行网格外推回退）
        jilu_b = board_results[center_board_idx][2]
        if len(jilu_b) > 0:
            print(f"  使用 get_mark_cord 检测到的X角点: {jilu_b}")
        else:
            print(f"  get_mark_cord 未检测到X角点（图片可能无蝴蝶形黑白半圆标记），不标出X角点")
        center_detected = [rec for rec in jilu_b]
        center_jilu = jilu_b
        center_inferred = []

        # 遮挡角点不推断、不补全：内部结果与可视化均只保留实际检测点。
        print(f"  中央棋盘格 #{center_board_idx+1} (质量={center_qual:.1f}, "
              f"距图像中心={board_centroids[0][0]:.1f}px, 范围 col[0,{full_col_max}] row[0,{full_row_max}]): "
              f"检测到{len(center_detected)}个X角点, 推断{len(center_inferred)}个缺失角点, 共{len(center_jilu)}个")
    else:
        print(f"  未找到可作为中央棋盘格的候选棋盘格")
    # ========================================================================

    # ---------- 可视化 ----------
    def _mask_points_near_boards(pts, boards, max_dist=80.0):  # 增大显示半径，确保棋盘格周边鞍点可见
        """保留距离任一已检测棋盘格网格点不超过 max_dist 像素的候选点"""
        if not boards or len(pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        board_pts = np.vstack([im for (im, _, _, _, _) in boards if len(im) > 0])
        if len(board_pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        tree = KDTree(board_pts)
        dist, _ = tree.query(pts, k=1)
        return dist <= max_dist

    # “鞍点”只显示真正归属于某个棋盘网格的检测响应，不再显示
    # 棋盘附近的任意角点候选。插值点不再作为白色网格点或连线节点。
    board_detected = np.zeros(len(points), dtype=bool)
    confirmed_boards = []
    if len(points) > 0:
        for im, wo, _, _, _ in final_two:
            if len(im) == 0:
                confirmed_boards.append((np.empty((0, 2)), np.empty((0, 2))))
                continue
            board_tree = KDTree(im)
            dist, node_idx = board_tree.query(points, k=1)
            mask = dist <= 2.5
            point_idx = np.where(mask)[0]
            # One original detection per board node; keep the closest match.
            best_by_node = {}
            for pi, ni, di in zip(point_idx, node_idx[mask], dist[mask]):
                ni = int(ni)
                if ni not in best_by_node or di < best_by_node[ni][1]:
                    best_by_node[ni] = (int(pi), float(di))
            selected_nodes = sorted(best_by_node)
            selected_points = [best_by_node[ni][0] for ni in selected_nodes]
            board_detected[selected_points] = True
            confirmed_boards.append((points[selected_points], wo[selected_nodes]))
    else:
        confirmed_boards = [(np.empty((0, 2)), np.empty((0, 2))) for _ in final_two]

    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    ax.imshow(img_gray, cmap='gray')
    ax.plot(points[board_detected, 0], points[board_detected, 1], 'yo', markersize=3,
            alpha=0.4, label=f'棋盘鞍点({np.sum(board_detected)}/{len(points)})')
    for idx, (im, wo) in enumerate(confirmed_boards):
        draw_chessboard_grid(ax, im, wo, img_shape=img_gray.shape)
        ax.plot(im[:, 0], im[:, 1], 'w+', markersize=9, markeredgewidth=1.6,
                label=f'棋盘格{idx + 1}', zorder=8)

    # 绘制所有棋盘格检测到的X角点（绿色实心大圆），推断的不标出
    if len(all_detected) > 0:
        first_label = True
        for rec in all_detected:
            mt, cx, cy = int(rec[0]), rec[1], rec[2]
            ax.plot(cx, cy, 'go', markersize=14, markeredgewidth=2.5,
                    alpha=0.9, zorder=10, label='X角点' if first_label else None)
            ax.text(cx + 8, cy - 8, str(mt), color='#00ff00', fontsize=11,
                    weight='bold', zorder=11,
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
            first_label = False

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    ax.set_title(f'{os.path.basename(img_path)}\n红=竖向边  蓝=横向边  绿圆=X角点  白十字=网格点')
    ax.axis('off')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(img_path))[0] + '_detect.png'
        plt.savefig(os.path.join(save_dir, out_name), dpi=160, bbox_inches='tight')
        # 额外保存调试图：青色=精化后评分前，黄色=评分通过，白色十字=棋盘格点
        fig_dbg, ax_dbg = plt.subplots(figsize=(14, 12))
        ax_dbg.imshow(img_gray, cmap='gray')
        if 'refined_p' in locals() and len(refined_p) > 0:
            ax_dbg.plot(refined_p[:, 0], refined_p[:, 1], 'co', markersize=2,
                        alpha=0.25, label=f'精化后({len(refined_p)})')
        ax_dbg.plot(points[board_detected, 0], points[board_detected, 1], 'yo', markersize=3,
                    alpha=0.5, label=f'棋盘归属({np.sum(board_detected)}/{len(points)})')
        for idx, (im, wo) in enumerate(confirmed_boards):
            ax_dbg.plot(im[:, 0], im[:, 1], 'w+', markersize=10, markeredgewidth=2.0,
                        label=f'棋盘格{idx + 1}({len(im)})', zorder=8)
        # 调试图中也加入X角点标记（所有检测到的）
        if len(all_detected) > 0:
            for rec in all_detected:
                mt, cx, cy = int(rec[0]), rec[1], rec[2]
                ax_dbg.plot(cx, cy, 'go', markersize=10, markeredgewidth=2.0,
                            alpha=0.85, zorder=10)
        ax_dbg.legend(loc='upper right')
        ax_dbg.set_title(f'{os.path.basename(img_path)} 调试视图（青=精化后/黄=评分通过/绿圆=X角点）')
        ax_dbg.axis('off')
        dbg_name = os.path.splitext(os.path.basename(img_path))[0] + '_debug.png'
        plt.savefig(os.path.join(save_dir, dbg_name), dpi=200, bbox_inches='tight')
        plt.close(fig_dbg)
    if show:
        plt.show()
        time.sleep(2)
    plt.close()
    return is_all_consistent, img_path


if False and __name__ == '__main__':
    img_folder = r'C:\Users\30772\Documents\2022nov1111'
    supported_ext = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
    img_list = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                if f.lower().endswith(supported_ext)]
    img_list.sort()
    if not img_list:
        print(f"文件夹 {img_folder} 下未找到图片")
        sys.exit(1)
    print(f"找到 {len(img_list)} 张图片，开始处理...")
    for img_path in img_list[0:]:
        is_consistent, path = process_single_image(img_path)

        # ========================================
        # ### 修改部分 2：移除主循环中的程序中止逻辑 ###
        # ========================================
        # 即使 is_consistent 为 False，也只打印提示，不调用 sys.exit(1)
        if not is_consistent:
            print(f"\n{'!' * 60}")
            print(f"警告: 旋向不一致，跳过当前图片继续处理: {os.path.basename(path)}")
            print(f"{'!' * 60}")
        # ========================================
        # ### 修改部分 2 结束 ###
        # ========================================

    print(f"\n{'=' * 60}")
    print("所有图片处理完毕。")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='棋盘位图高精度识别与调试图输出')
    parser.add_argument('--img-folder', default=r'D:\棋盘\2022nov1111',
                        help='2022nov1111')
    parser.add_argument('--limit', type=int, default=0, help='最多处理多少张，0 表示全部')
    parser.add_argument('--start', type=int, default=0, help='从排序后的第几张开始处理')
    parser.add_argument('--no-show', action='store_true', help='不弹出 Matplotlib 窗口')
    parser.add_argument('--save-dir', default=os.path.join(SCRIPT_DIR, '识别结果'),
                        help='调试图保存目录')
    args = parser.parse_args()

    img_folder = args.img_folder
    supported_ext = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
    img_list = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                if f.lower().endswith(supported_ext)]
    img_list.sort()
    img_list = img_list[args.start:]
    if args.limit > 0:
        img_list = img_list[:args.limit]
    if not img_list:
        print(f"文件夹 {img_folder} 下未找到图片")
        sys.exit(1)
    print(f"找到 {len(img_list)} 张图片，开始处理...")
    for img_path in img_list:
        is_consistent, path = process_single_image(
            img_path,
            show=not args.no_show,
            save_dir=args.save_dir
        )
        if not is_consistent:
            print(f"\n{'!' * 60}")
            print(f"警告: 旋向不一致，跳过当前图片继续处理: {os.path.basename(path)}")
            print(f"{'!' * 60}")

    print(f"\n{'=' * 60}")
    print("所有图片处理完毕。")
    print(f"{'=' * 60}")
