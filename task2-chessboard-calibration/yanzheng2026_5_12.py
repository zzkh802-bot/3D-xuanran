import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
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
    max_residual = 5.0  # 最大容忍残差（像素）
    score = max(0.0, 1.0 - residual / max_residual)

    return score


def evaluate_local_grid_alignment(new_local_pt, anchor_local_pt, grid_dir, ref_len=0):
    """
    Score whether a new point stays on the expected local grid line.
    移除了未使用的 existing_local_pts 参数，直接基于锚点评估垂直偏移与步长。
    """
    perpendicular_axis = 1 if grid_dir == 0 else 0
    travel_axis = 0 if grid_dir == 0 else 1
    tolerance = max(2.5, ref_len * 0.16) if ref_len > 0 else 3.0

    perp_error = abs(new_local_pt[perpendicular_axis] - anchor_local_pt[perpendicular_axis])
    perp_score = np.exp(-0.5 * (perp_error / (tolerance + 1e-12)) ** 2)

    if ref_len <= 0:
        return float(perp_score)

    step = abs(new_local_pt[travel_axis] - anchor_local_pt[travel_axis])
    step_error = abs(step - ref_len)
    step_score = np.exp(-0.5 * (step_error / (ref_len * 0.22 + 1e-12)) ** 2)
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
    # 方向约束：夹角小（cos > 0.95）且沿正方向
    mask_dir = (cos_angle > 0.95) & (proj > 0)
    # 排除自身和已使用点
    mask_avail = ~used_mask
    mask_avail[center_idx] = False
    mask = mask_dir & mask_avail
    if ref_len > 0:
        ratio = dists / (ref_len + 1e-12)
        mask &= (ratio > 0.7) & (ratio < 1.35)
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
            collinearity_thresh = 0.52
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
                points, dirs, quad_idx, max_iter=80, dist_thresh=5.0
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
    primary = chessboardsgrow_py(points, directions, scores, img_shape, max_boards=max_boards)
    legacy = legacy_homography_chessboardsgrow_py(
        points, directions, scores, img_shape, max_boards=max_boards
    )
    if legacy:
        print(f"  旧单应生长候选 {len(legacy)} 个")
    return primary + legacy


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


def _trim_sparse_border_grid(img_pts, world_pts, min_fill=0.45, min_points=9):
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

        if len(matched_grid) < max(min_points, int(len(img_pts) * 0.82)):
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
    for axis in (0, 1):
        trial_units = units.copy()
        trial_units[:, axis] *= 2.0
        H_img_to_grid = compute_homography(img_pts, trial_units)
        if H_img_to_grid is None:
            continue
        projected = _project_world_points(H_img_to_grid, unused)
        if projected is None:
            continue
        max_units = np.max(trial_units, axis=0)
        close_count = 0
        errors = []
        for point in projected:
            if not np.isfinite(point).all():
                continue
            rounded = np.round(point).astype(int)
            if np.any(rounded < -1) or rounded[0] > max_units[0] + 1 or rounded[1] > max_units[1] + 1:
                continue
            if rounded[axis] % 2 != 1:
                continue
            err = float(np.linalg.norm(point - rounded))
            if err < 0.28:
                close_count += 1
                errors.append(err)
        if close_count >= max(min_inside, int(len(unused) * 0.35)):
            score = close_count / (np.median(errors) + 1e-6)
            if best is None or score > best[0]:
                best = (score, axis, trial_units)

    if best is None:
        return img_pts, world_pts, compute_homography(world_pts, img_pts)

    _, axis, adjusted_units = best
    adjusted_units = adjusted_units - np.min(adjusted_units, axis=0)
    adjusted_world = adjusted_units * 15.0
    print(f"  跳格修正: {'X/列' if axis == 0 else 'Y/行'} 轴加密")
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
    img_s, world_s = sort_chessboard_points(img_pts, world_pts)
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


def _filter_points_inside_polygon(img_pts, world_pts, ref_img_pts):
    """
    过滤掉落在 ref_img_pts 凸包外的点。
    用于防止棋盘格重建/插值时把外围候选点误拉进可视化网格。
    """
    if len(ref_img_pts) < 3 or len(img_pts) == 0:
        return img_pts, world_pts
    poly, _ = get_chessboard_polygon(ref_img_pts)
    if poly is None:
        return img_pts, world_pts
    inside = poly.contains_points(img_pts)
    if np.sum(inside) < 4:
        return img_pts, world_pts
    return img_pts[inside], world_pts[inside]


def _filter_false_positive_x_corners(jilu_b, img_b, min_dist_ratio=0.35):
    """
    过滤掉落在棋盘格普通网格角点上的假阳性 X 角点。

    真正的蝴蝶形黑白半圆 X 角点位于棋盘格网格外侧半格处，
    距离最近的棋盘格网格点约为半格间距；而假阳性通常是普通
    棋盘格角点，距离网格点非常近。
    """
    if len(jilu_b) == 0 or len(img_b) < 2:
        return jilu_b

    tree = KDTree(img_b)
    dists, _ = tree.query(img_b, k=2)
    median_spacing = float(np.median(dists[:, 1]))
    min_dist = max(5.0, min_dist_ratio * median_spacing)

    filtered = []
    for rec in jilu_b:
        cx, cy = float(rec[1]), float(rec[2])
        d, _ = tree.query([[cx, cy]], k=1)
        if d[0] >= min_dist:
            filtered.append(rec)
        else:
            print(f"  [过滤] 疑似假阳性 X 角点 (type={int(rec[0])}, "
                  f"位置=({cx:.1f},{cy:.1f})): 距最近网格点 {d[0]:.1f}px < {min_dist:.1f}px")
    return filtered


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
    # ========== 替换为 MATLAB 风格棋盘格生长 ==========
    points = final_points.copy()
    # 替换原来的调用
    dirs = np.column_stack([final_v1, final_v2])
    found_chessboards_raw = find_chessboard_candidates(points, dirs, final_scores, img_gray.shape, max_boards=6)
    if not found_chessboards_raw:
        print("未找到有效棋盘格")
        return False, img_path
    # 将 raw 结果转换为原有格式：质量评分 + 重叠过滤
    scored_boards = []
    for img_pts, world_pts, H, used_idx in found_chessboards_raw:
        img_pts, world_pts, H = prepare_chessboard_candidate(img_pts, world_pts, final_points, H)
        # 世界坐标原点归零
        world_pts = world_pts - np.min(world_pts, axis=0)
        # 网格规整度验证
        valid, med_dev = validate_chessboard_grid_alignment(world_pts)
        if not valid:
            print(f"  [X] 丢弃棋盘格 (偏差 {med_dev:.2f})")
            continue
        quality = calculate_chessboard_quality(img_pts, world_pts)
        p90_line, worst_line, bad_groups = chessboard_line_residual_summary(img_pts, world_pts)
        print(
            f"  候选棋盘: 点数 {len(img_pts)}, 线残差p90 {p90_line:.2f}, 最大残差 {worst_line:.2f}, 异常组 {bad_groups}, 质量 {quality:.2f}")
        scored_boards.append((img_pts, world_pts, H, used_idx, quality))
    # 按质量降序，取不重叠的棋盘格（数量不限，后续由MAX_BOARDS控制总量）
    # ⚠ 重要：用图像空间重叠检查，而非世界坐标（世界坐标已归零，不同棋盘格可能
    #    巧合重叠）
    MAX_BOARDS = 6  # 提升：从3→6，识别更多棋盘格
    PROPAGATION_MIN_QUALITY = 5.0  # 传播新增棋盘格的最低质量阈值
    scored_boards.sort(key=lambda x: x[4], reverse=True)
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
            for src_idx, (src_img, src_world, src_H, _, _) in enumerate(final_two):
                propagated = propagate_chessboard_by_homography(
                    src_img, src_world, src_H, final_points,
                    grid_spacing=15.0, min_points=6
                )
                if propagated is not None:
                    p_img, p_world, p_H, _ = propagated
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
        # 保存重建前的原始点，用于剔除外扩/插值引入的凸包外点
        orig_img = img_pts.copy()
        orig_world = world_pts.copy()

        # 关闭外扩 margin：只补当前 bbox 内部的缺失点，避免把外围候选点拉进来
        clean_img, clean_world, clean_H = complete_chessboard_from_candidates(
            img_pts, world_pts, final_points, expand_margin=0)
        # 过滤重建阶段引入的凸包外点
        clean_img, clean_world = _filter_points_inside_polygon(clean_img, clean_world, orig_img)

        before_fill = len(clean_img)
        # 根据当前完整度决定插值策略：高完整度棋盘格直接补齐 bbox 内所有缺失网格点，
        # 避免细长棋盘格的顶角/边角因邻居不足而遗漏；低完整度棋盘格仍保守插值。
        snapped = np.round(clean_world / 15.0).astype(int)
        grid_cells = len(np.unique(snapped[:, 0])) * len(np.unique(snapped[:, 1]))
        fill_ratio = len(clean_img) / max(1, grid_cells)
        min_n = 0 if fill_ratio > 0.70 else 2
        clean_img, clean_world = interpolate_missing_grid_points(clean_img, clean_world, min_neighbors=min_n)
        # 注：插值得到的是规则网格单应投影点，不再用原始凸包过滤，避免边界缺失点被误删。

        if len(clean_img) != before_fill:
            print(f"  网格插值填补: {before_fill} -> {len(clean_img)} 个交叉点")
            clean_H = compute_homography(clean_world, clean_img)
        cleaned_final_two.append((clean_img, clean_world, clean_H if clean_H is not None else H, used_idx, qual))

    final_two = cleaned_final_two
    # 提取各棋盘格数据（支持任意数量）
    board_results = []  # [(xy, uv, jilu), ...]
    for (img_b, world_b, H_b, _, _) in final_two:
        # 强制归一化为参考代码格式：2 列竖直、x∈[0,15]、y 向下增长、按行排序
        world_n = world_b - np.min(world_b, axis=0)
        max_r = np.max(world_n, axis=0)
        if max_r[0] > max_r[1]:
            world_n = world_n[:, [1, 0]]
            world_n[:, 0] = 15.0 - world_n[:, 0]
        order = np.lexsort((world_n[:, 0], world_n[:, 1]))
        world_n = world_n[order]
        img_n = img_b[order]
        H_n = compute_homography(world_n, img_n)

        xy_b = np.column_stack([world_n, np.ones(len(world_n))])
        uv_b = np.column_stack([img_n, np.ones(len(img_n))])
        xy_b, uv_b, jilu_b = get_mark_cord(
            final_points, corner, xy_b, uv_b, H_n,
            nn, spij3, spij7, [], xcsp, Im0, mm, img_gray)
        # 过滤普通棋盘格角点被误标为 X 角点的情况
        jilu_b = _filter_false_positive_x_corners(jilu_b, img_n)
        board_results.append((xy_b, uv_b, jilu_b))


    # 收集所有棋盘格实际检测到的X角点（不仅限于中央棋盘格，用于可视化）
    all_detected = []
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

        # 不足4个X角点：推断缺失的（用于内部 H_corner 等计算，但推断的不标出）
        if 0 < len(jilu_b) < 4:
            jilu_full = _infer_missing_corners_from_partial(
                jilu_b, center_wo, center_im, grid_spacing=15.0)
            if jilu_full is not None and len(jilu_full) == 4:
                center_jilu = jilu_full
                detected_ids = {int(r[0]) for r in jilu_b}
                center_inferred = [r for r in jilu_full if int(r[0]) not in detected_ids]
        print(f"  中央棋盘格 #{center_board_idx+1} (质量={center_qual:.1f}, "
              f"距图像中心={board_centroids[0][0]:.1f}px, 范围 col[0,{full_col_max}] row[0,{full_row_max}]): "
              f"检测到{len(center_detected)}个X角点, 推断{len(center_inferred)}个缺失角点, 共{len(center_jilu)}个")
    else:
        print(f"  未找到可作为中央棋盘格的候选棋盘格")
    # ========================================================================

    # ---------- 可视化 ----------
    def _mask_points_near_boards(pts, boards, max_dist=40.0):
        """保留距离任一已检测棋盘格网格点不超过 max_dist 像素的候选点"""
        if not boards or len(pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        board_pts = np.vstack([im for (im, _, _, _, _) in boards if len(im) > 0])
        if len(board_pts) == 0:
            return np.zeros(len(pts), dtype=bool)
        tree = KDTree(board_pts)
        dist, _ = tree.query(pts, k=1)
        return dist <= max_dist

    near_board = _mask_points_near_boards(points, final_two, max_dist=40.0)

    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    ax.imshow(img_gray, cmap='gray')
    ax.plot(points[near_board, 0], points[near_board, 1], 'yo', markersize=3,
            alpha=0.4, label=f'鞍点({np.sum(near_board)}/{len(points)})')
    for idx, (im, wo, _, _, _) in enumerate(final_two):
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
        ax_dbg.plot(points[near_board, 0], points[near_board, 1], 'yo', markersize=3,
                    alpha=0.5, label=f'评分通过({np.sum(near_board)}/{len(points)})')
        for idx, (im, wo, _, _, _) in enumerate(final_two):
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
