"""
棋盘格与标记点联合检测（最终修复版）
核心修复：1. 移除错误的世界坐标归一化 2. 修复M2索引赋值
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import time

# ======================== 图像读取 ========================
def imread_gray(path):
    img = plt.imread(path)
    img = np.flipud(img)
    if img.ndim == 3:
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float64) / 65535.0
    else:
        img = img.astype(np.float64)
    return img

# ======================== 梯度计算 ========================
def compute_gradients(img):
    du = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    dv = du.T
    img_du = convolve2d(img, du, mode='same', boundary='symm')
    img_dv = convolve2d(img, dv, mode='same', boundary='symm')
    magnitude = np.sqrt(img_du**2 + img_dv**2)
    angle = np.arctan2(img_dv, img_du)
    angle[angle < 0] += np.pi
    angle[angle >= np.pi] -= np.pi
    return angle, magnitude, img_du, img_dv

# ======================== 四象限模板 ========================
def create_correlation_patch(angle_1, angle_2, radius):
    width = height = 2 * radius + 1
    mu = mv = radius
    n1 = np.array([-np.sin(angle_1), np.cos(angle_1)])
    n2 = np.array([-np.sin(angle_2), np.cos(angle_2)])
    sigma = radius / 2.0
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    dist = np.sqrt(x*x + y*y)
    gauss = np.exp(-0.5 * (dist / sigma)**2) / (sigma * np.sqrt(2*np.pi))
    gauss /= gauss.sum()
    a1 = np.zeros((height, width))
    a2 = np.zeros((height, width))
    b1 = np.zeros((height, width))
    b2 = np.zeros((height, width))
    for u in range(width):
        for v in range(height):
            vec = np.array([u - mu, v - mv])
            s1 = np.dot(vec, n1)
            s2 = np.dot(vec, n2)
            w = gauss[v, u]
            if s1 <= -0.1 and s2 <= -0.1:
                a1[v, u] = w
            elif s1 >= 0.1 and s2 >= 0.1:
                a2[v, u] = w
            elif s1 <= -0.1 and s2 >= 0.1:
                b1[v, u] = w
            elif s1 >= 0.1 and s2 <= -0.1:
                b2[v, u] = w
    for mat in (a1, a2, b1, b2):
        s = mat.sum()
        if s > 0:
            mat /= s
    return a1, a2, b1, b2

# ======================== 多尺度模板滤波 ========================
def filter_image_with_templates_fft(img, radius_list):
    height, width = img.shape
    corner_map = np.zeros((height, width))
    template_props = [
        (0, np.pi/2, radius_list[0]),
        (np.pi/4, -np.pi/4, radius_list[0]),
        (0, np.pi/2, radius_list[1]),
        (np.pi/4, -np.pi/4, radius_list[1]),
        (0, np.pi/2, radius_list[2]),
        (np.pi/4, -np.pi/4, radius_list[2])
    ]
    for ang1, ang2, r in template_props:
        a1, a2, b1, b2 = create_correlation_patch(ang1, ang2, r)
        img_a1 = fftconvolve(img, a1, mode='same')
        img_a2 = fftconvolve(img, a2, mode='same')
        img_b1 = fftconvolve(img, b1, mode='same')
        img_b2 = fftconvolve(img, b2, mode='same')
        mu = (img_a1 + img_a2 + img_b1 + img_b2) * 0.25
        response1 = np.minimum(np.minimum(img_a1 - mu, img_a2 - mu),
                               np.minimum(mu - img_b1, mu - img_b2))
        response2 = np.minimum(np.minimum(mu - img_a1, mu - img_a2),
                               np.minimum(img_b1 - mu, img_b2 - mu))
        corner_map = np.maximum(corner_map, response1)
        corner_map = np.maximum(corner_map, response2)
    return corner_map

# ======================== 非极大值抑制 ========================
def non_maximum_suppression_fast(corner_map, n=3, tau=0.025, margin=5):
    height, width = corner_map.shape
    maxima_mask = np.zeros_like(corner_map, dtype=bool)
    step = n + 1
    for i in range(n+1+margin, width-n-margin, step):
        for j in range(n+1+margin, height-n-margin, step):
            i_end = min(i+step, width)
            j_end = min(j+step, height)
            block = corner_map[j:j_end, i:i_end]
            max_val = block.max()
            if max_val < tau:
                continue
            max_pos = np.unravel_index(np.argmax(block), block.shape)
            max_i = i + max_pos[1]
            max_j = j + max_pos[0]
            y_min = max(0, max_j - n)
            y_max = min(height, max_j + n + 1)
            x_min = max(0, max_i - n)
            x_max = min(width, max_i + n + 1)
            surround = corner_map[y_min:y_max, x_min:x_max]
            if corner_map[max_j, max_i] == surround.max():
                maxima_mask[max_j, max_i] = True
    maxima_mask[:margin, :] = False
    maxima_mask[-margin:, :] = False
    maxima_mask[:, :margin] = False
    maxima_mask[:, -margin:] = False
    y_idx, x_idx = np.where(maxima_mask)
    return np.column_stack((x_idx, y_idx)).astype(np.float64)

# ======================== 均值漂移与边缘方向 ========================
def find_modes_mean_shift(hist, sigma=1.0):
    n_bins = len(hist)
    if n_bins == 0:
        return np.array([])
    kernel_radius = int(round(2 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-kernel_radius, kernel_radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    hist_smoothed = np.convolve(hist, kernel, mode='same')
    modes = []
    for i in range(n_bins):
        curr = i
        while True:
            left = (curr - 1) % n_bins
            right = (curr + 1) % n_bins
            h0, h_left, h_right = hist_smoothed[curr], hist_smoothed[left], hist_smoothed[right]
            if h_left >= h0 and h_left >= h_right:
                curr = left
            elif h_right > h0 and h_right > h_left:
                curr = right
            else:
                break
        if modes and curr in np.array(modes)[:, 0]:
            continue
        modes.append([curr, hist_smoothed[curr]])
    if not modes:
        return np.array([])
    modes = np.array(modes)
    return modes[np.argsort(modes[:, 1])[::-1]]

def edge_orientations_fast(img_angle_sub, img_weight_sub, bin_num=32):
    vec_angle = img_angle_sub.ravel()
    vec_weight = img_weight_sub.ravel()
    vec_angle = vec_angle + np.pi/2
    vec_angle[vec_angle > np.pi] -= np.pi
    bins = np.linspace(0, np.pi, bin_num+1)
    hist, _ = np.histogram(vec_angle, bins=bins, weights=vec_weight)
    modes = find_modes_mean_shift(hist, 1.0)
    if modes.shape[0] < 2:
        return np.zeros(2), np.zeros(2)
    bin_width = np.pi / bin_num
    mode_angles = modes[:2, 0] * bin_width
    mode_angles = np.sort(mode_angles)
    delta = min(mode_angles[1] - mode_angles[0], mode_angles[0] + np.pi - mode_angles[1])
    if delta <= 0.3:
        return np.zeros(2), np.zeros(2)
    v1 = np.array([np.cos(mode_angles[0]), np.sin(mode_angles[0])])
    v2 = np.array([np.cos(mode_angles[1]), np.sin(mode_angles[1])])
    return v1, v2

# ======================== 亚像素精化 ========================
def refine_corners_vectorized(img_du, img_dv, img_angle, img_weight, corners, r=10, img_gray=None):
    height, width = img_du.shape
    refined_p = []
    refined_v1 = []
    refined_v2 = []
    grad_norm = np.sqrt(img_du**2 + img_dv**2)
    grad_norm[grad_norm < 0.1] = 1.0
    u_norm = img_du / grad_norm
    v_norm = img_dv / grad_norm
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    for (cu, cv) in corners:
        cu_int, cv_int = int(round(cu)), int(round(cv))
        ymin = max(cv_int - r, 0)
        ymax = min(cv_int + r + 1, height)
        xmin = max(cu_int - r, 0)
        xmax = min(cu_int + r + 1, width)
        if img_gray is not None:
            patch_gray = img_gray[ymin:ymax, xmin:xmax]
            if patch_gray.max() - patch_gray.min() < 0.03:
                continue
        angle_sub = img_angle[ymin:ymax, xmin:xmax]
        weight_sub = img_weight[ymin:ymax, xmin:xmax]
        v1, v2 = edge_orientations_fast(angle_sub, weight_sub)
        if np.all(v1 == 0) or np.all(v2 == 0):
            continue
        u_loc = u_norm[ymin:ymax, xmin:xmax]
        v_loc = v_norm[ymin:ymax, xmin:xmax]
        dot1 = u_loc * v1[0] + v_loc * v1[1]
        dot2 = u_loc * v2[0] + v_loc * v2[1]
        inlier1 = np.abs(dot1) < 0.25
        inlier2 = np.abs(dot2) < 0.25
        A1 = np.zeros((2,2))
        A2 = np.zeros((2,2))
        if np.any(inlier1):
            u1 = u_loc[inlier1]
            v1_ = v_loc[inlier1]
            A1[0,0] = np.sum(u1**2)
            A1[0,1] = A1[1,0] = np.sum(u1 * v1_)
            A1[1,1] = np.sum(v1_**2)
        if np.any(inlier2):
            u2 = u_loc[inlier2]
            v2_ = v_loc[inlier2]
            A2[0,0] = np.sum(u2**2)
            A2[0,1] = A2[1,0] = np.sum(u2 * v2_)
            A2[1,1] = np.sum(v2_**2)
        if np.linalg.matrix_rank(A1) == 2:
            eigvals, eigvecs = np.linalg.eigh(A1)
            v1 = eigvecs[:, 0]
        if np.linalg.matrix_rank(A2) == 2:
            eigvals, eigvecs = np.linalg.eigh(A2)
            v2 = eigvecs[:, 0]
        yy = y_grid[ymin:ymax, xmin:xmax]
        xx = x_grid[ymin:ymax, xmin:xmax]
        dy = yy - cv_int
        dx = xx - cu_int
        w = np.stack([dx, dy], axis=-1)
        proj1 = (w[...,0]*v1[0] + w[...,1]*v1[1])[..., np.newaxis] * v1
        dist1 = np.linalg.norm(w - proj1, axis=-1)
        proj2 = (w[...,0]*v2[0] + w[...,1]*v2[1])[..., np.newaxis] * v2
        dist2 = np.linalg.norm(w - proj2, axis=-1)
        cond1 = (dist1 < 3) & (np.abs(dot1) < 0.25)
        cond2 = (dist2 < 3) & (np.abs(dot2) < 0.25)
        valid_mask = cond1 | cond2
        center_y = cv_int - ymin
        center_x = cu_int - xmin
        if 0 <= center_y < valid_mask.shape[0] and 0 <= center_x < valid_mask.shape[1]:
            valid_mask[center_y, center_x] = False
        if not np.any(valid_mask):
            continue
        u_val = u_loc[valid_mask]
        v_val = v_loc[valid_mask]
        x_val = xx[valid_mask]
        y_val = yy[valid_mask]
        H_xx = u_val**2
        H_xy = u_val * v_val
        H_yy = v_val**2
        G = np.array([[H_xx.sum(), H_xy.sum()],
                      [H_xy.sum(), H_yy.sum()]])
        b_x = np.sum(H_xx * x_val + H_xy * y_val)
        b_y = np.sum(H_xy * x_val + H_yy * y_val)
        b = np.array([b_x, b_y])
        if np.linalg.matrix_rank(G) == 2:
            try:
                new_pos = np.linalg.solve(G, b)
                if np.linalg.norm(new_pos - np.array([cu, cv])) < 4:
                    cu, cv = new_pos
                else:
                    continue
            except:
                continue
        else:
            continue
        refined_p.append((cu, cv))
        refined_v1.append(v1)
        refined_v2.append(v2)
    return np.array(refined_p), np.array(refined_v1), np.array(refined_v2)

# ======================== 评分 ========================
def corner_correlation_score(patch_gray, patch_mag, v1, v2):
    c = (np.array(patch_gray.shape) - 1) / 2
    r = int(c[0])
    filt = -np.ones_like(patch_mag)
    yy, xx = np.mgrid[0:2*r+1, 0:2*r+1]
    p1 = np.stack([xx - c[0], yy - c[1]], axis=-1)
    proj1 = np.sum(p1 * v1, axis=-1, keepdims=True) * v1
    dist1 = np.linalg.norm(p1 - proj1, axis=-1)
    proj2 = np.sum(p1 * v2, axis=-1, keepdims=True) * v2
    dist2 = np.linalg.norm(p1 - proj2, axis=-1)
    mask = (dist1 <= 1.5) | (dist2 <= 1.5)
    filt[mask] = 1.0
    vec_w = patch_mag.ravel()
    vec_f = filt.ravel()
    vec_w = (vec_w - vec_w.mean()) / (vec_w.std() + 1e-8)
    vec_f = (vec_f - vec_f.mean()) / (vec_f.std() + 1e-8)
    score_grad = max(np.dot(vec_w, vec_f) / (len(vec_w)-1), 0)
    a1, a2, b1, b2 = create_correlation_patch(np.arctan2(v1[1], v1[0]),
                                              np.arctan2(v2[1], v2[0]), r)
    s_a1 = np.sum(a1 * patch_gray)
    s_a2 = np.sum(a2 * patch_gray)
    s_b1 = np.sum(b1 * patch_gray)
    s_b2 = np.sum(b2 * patch_gray)
    mu = (s_a1 + s_a2 + s_b1 + s_b2) / 4.0
    score1 = min(s_a1 - mu, s_a2 - mu, mu - s_b1, mu - s_b2)
    score2 = min(mu - s_a1, mu - s_a2, s_b1 - mu, s_b2 - mu)
    score_intensity = max(score1, score2, 0)
    return score_grad * score_intensity

def score_corners_batch(img, img_weight, corners_p, corners_v1, corners_v2, radius_list):
    scores = []
    h, w = img.shape
    for (x, y), v1, v2 in zip(corners_p, corners_v1, corners_v2):
        best = 0.0
        xi, yi = int(round(x)), int(round(y))
        for r in radius_list:
            if xi < r or xi >= w-r or yi < r or yi >= h-r:
                continue
            patch_gray = img[yi-r:yi+r+1, xi-r:xi+r+1]
            patch_mag = img_weight[yi-r:yi+r+1, xi-r:xi+r+1]
            sc = corner_correlation_score(patch_gray, patch_mag, v1, v2)
            if sc > best:
                best = sc
        scores.append(best)
    return np.array(scores)

# ======================== 模糊评估与合并 ========================
def measure_blur_level(img):
    lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    lap_img = convolve2d(img, lap_kernel, mode='valid')
    return lap_img.var()

def merge_close_points(points, dirs, scores, dist_thresh=8.0, ratio=0.5):
    if len(points) < 2:
        return points, dirs
    order = np.argsort(scores)[::-1]
    pts_sorted = points[order]
    dirs_sorted = dirs[order]
    scores_sorted = scores[order]
    keep = np.ones(len(pts_sorted), dtype=bool)
    tree = cKDTree(pts_sorted)
    for i in range(len(pts_sorted)):
        if not keep[i]:
            continue
        idxs = tree.query_ball_point(pts_sorted[i], dist_thresh)
        for j in idxs:
            if j == i or not keep[j]:
                continue
            if scores_sorted[j] < scores_sorted[i] * ratio:
                keep[j] = False
    inv_order = np.argsort(order)
    keep = keep[inv_order]
    return points[keep], dirs[keep]

def adjust_orientation_handedness(v1, v2):
    n1 = np.column_stack([v1[:,1], -v1[:,0]])
    flip = -np.sign(n1[:,0]*v2[:,0] + n1[:,1]*v2[:,1])
    v2_adj = v2 * flip[:, np.newaxis]
    return v1, v2_adj

# ======================== 棋盘格生长 ========================
def get_little_four_P_py(points, directions, start_idx, img_shape):
    """
    🔥 严格版：初始小四边形检测 + 无报错修复版
    核心约束：四边形的两条邻边 必须 分别精准匹配 v1、v2 方向，否则判定失败
    统一返回 None, None，杜绝解包报错
    """
    h, w = img_shape[:2]
    start_pt = points[start_idx]
    v1 = directions[start_idx, 0]
    v2 = directions[start_idx, 1]

    # 单位化方向向量
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

    # 计算相对向量
    vecs = points - start_pt
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    vecs_norm[start_idx] = 0

    # 严格方向筛选：仅保留与v1/v2方向高度匹配的点
    dot_v1_all = np.abs(np.dot(vecs_norm, v1_norm))
    mask_v1 = (dot_v1_all >= 0.9) & (np.linalg.norm(vecs, axis=1) > 3)
    candidate_v1 = np.where(mask_v1)[0]

    dot_v2_all = np.abs(np.dot(vecs_norm, v2_norm))
    mask_v2 = (dot_v2_all >= 0.9) & (np.linalg.norm(vecs, axis=1) > 3)
    candidate_v2 = np.where(mask_v2)[0]

    # 无候选点 → 返回None, None
    if len(candidate_v1) == 0 or len(candidate_v2) == 0:
        return None, None

    idx1 = candidate_v1[np.argmax(dot_v1_all[candidate_v1])]
    idx2 = candidate_v2[np.argmax(dot_v2_all[candidate_v2])]

    p0 = start_pt
    p1 = points[idx1]
    p2 = points[idx2]

    # 预测第四个点
    p3_pred = p1 + (p2 - p0)
    p3_pred = np.round(p3_pred).astype(int)

    # 搜索p3
    search_win = 3
    found_idx = None
    for dy in range(-search_win, search_win+1):
        for dx in range(-search_win, search_win+1):
            cand = p3_pred + np.array([dx, dy])
            if 0 <= cand[0] < w and 0 <= cand[1] < h:
                dists = np.linalg.norm(points - cand, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < 3.0:
                    found_idx = min_idx
                    break
        if found_idx is not None:
            break
    # 未找到p3 → 返回None, None
    if found_idx is None:
        return None, None

    # 构建四边形
    quad_pts = np.array([p0, p1, points[found_idx], p2])
    quad_idx = [start_idx, idx1, found_idx, idx2]

    # 边长约束
    vecs_quad = np.diff(quad_pts, axis=0, append=quad_pts[:1])
    lens = np.linalg.norm(vecs_quad, axis=1)
    if np.any(lens > 100) or np.min(lens) < 5:
        return None, None

    # 最终方向校验
    edge1 = quad_pts[1] - quad_pts[0]
    edge2 = quad_pts[3] - quad_pts[0]
    edge1_norm = edge1 / (np.linalg.norm(edge1) + 1e-8)
    edge2_norm = edge2 / (np.linalg.norm(edge2) + 1e-8)

    match1 = np.abs(np.dot(edge1_norm, v1_norm)) >= 0.8
    match2 = np.abs(np.dot(edge2_norm, v2_norm)) >= 0.8
    if not (match1 and match2):
        return None, None

    # 左右手系修正
    vec1_cross = quad_pts[1] - quad_pts[0]
    vec2_cross = quad_pts[2] - quad_pts[0]
    det_val = vec1_cross[0] * vec2_cross[1] - vec1_cross[1] * vec2_cross[0]
    if det_val < 0:
        quad_pts[[1, 3]] = quad_pts[[3, 1]]
        quad_idx[1], quad_idx[3] = quad_idx[3], quad_idx[1]

    return quad_pts, quad_idx

def compute_homography(src_pts, dst_pts):
    """100%复刻MATLAB getHH1逻辑"""
    xy = np.asarray(src_pts, dtype=np.float64)
    abc1 = np.asarray(dst_pts, dtype=np.float64)
    N = xy.shape[0]
    xy_mean = np.mean(xy, axis=0)
    abc1_mean = np.mean(abc1, axis=0)
    s1 = np.sqrt(np.mean(np.sum((xy - xy_mean)**2, axis=1)))
    s2 = np.sqrt(np.mean(np.sum((abc1 - abc1_mean)**2, axis=1)))
    k = np.sqrt(2) / np.array([s1, s2])
    Ha1 = np.array([[k[0], 0, -k[0]*xy_mean[0]],
                    [0, k[0], -k[0]*xy_mean[1]],
                    [0, 0, 1]])
    Ha2 = np.array([[k[1], 0, -k[1]*abc1_mean[0]],
                    [0, k[1], -k[1]*abc1_mean[1]],
                    [0, 0, 1]])
    xy_h = np.hstack([xy, np.ones((N,1))])
    abc1_h = np.hstack([abc1, np.ones((N,1))])
    xy_n = (Ha1 @ xy_h.T).T
    abc1_n = (Ha2 @ abc1_h.T).T
    A = np.zeros((2*N, 9))
    for i in range(N):
        X, Y, W = xy_n[i]
        u, v, _ = abc1_n[i]
        A[2*i] = [X, Y, W, 0, 0, 0, -u*X, -u*Y, -u*W]
        A[2*i+1] = [0, 0, 0, X, Y, W, -v*X, -v*Y, -v*W]
    _, _, Vt = np.linalg.svd(A.T @ A)
    H1 = Vt[-1].reshape(3, 3, order='F').T
    HH0 = np.linalg.inv(Ha2) @ H1 @ Ha1
    if np.sum(np.sign((HH0 @ xy_h.T).T[:,2])) < 0:
        HH0 = -HH0
    HH0 /= HH0[2, 2]
    return HH0

def grow_chessboard_region(points, directions, start_quad_idx, max_iter=30, dist_thresh=3.0):
    world_pts = np.array([[0,0], [15,0], [15,15], [0,15]], dtype=np.float64)
    img_pts = points[list(start_quad_idx)]
    H = compute_homography(world_pts, img_pts)
    all_img = img_pts.copy()
    all_world = world_pts.copy()
    used_idx = set(start_quad_idx)
    pts_tree = cKDTree(points)
    for _ in range(max_iter):
        min_xy = np.min(all_world, axis=0) // 15
        max_xy = np.max(all_world, axis=0) // 15
        candidates_world = []
        for x in range(int(min_xy[0])-1, int(max_xy[0])+2):
            for y in range(int(min_xy[1])-1, int(max_xy[1])+2):
                wpt = np.array([x*15, y*15])
                if not any(np.all(np.isclose(all_world, wpt), axis=1)):
                    candidates_world.append(wpt)
        if not candidates_world:
            break
        candidates_world = np.array(candidates_world)
        ones = np.ones((len(candidates_world), 1))
        proj = H @ np.hstack([candidates_world, ones]).T
        proj = (proj[:2] / proj[2]).T
        added = False
        for i, wpt in enumerate(candidates_world):
            pred_pt = proj[i]
            dist, min_idx = pts_tree.query(pred_pt)
            if dist < dist_thresh and min_idx not in used_idx:
                all_img = np.vstack([all_img, points[min_idx]])
                all_world = np.vstack([all_world, wpt])
                used_idx.add(min_idx)
                added = True
        if not added:
            break
        H = compute_homography(all_world, all_img)
    unique_x = np.unique(np.round(all_world[:, 0], decimals=1))
    unique_y = np.unique(np.round(all_world[:, 1], decimals=1))
    return all_img, all_world, H, used_idx

# ======================== 标记点识别（修复M2赋值） ========================
# ======================== 标记点识别（最终修复：加边界检查） ========================
def get_mark_cord(xc, corner, xy1, uv1, HH0, nn, spij3, spij7, jilu, xcsp, Im0, mm):
    h, w = nn
    max_index = h * w  # 图像最大线性索引

    xy_add = np.array([
        [7.5, -51.5, 1], [7.5, -66.5, 1], [7.5, -81.5, 1],
        [7.5, 186.5, 1], [7.5, 171.5, 1], [7.5, 156.5, 1],
        [7.5, 141.5, 1], [7.5, 126.5, 1], [7.5, 111.5, 1],
        [7.5, 96.5, 1], [7.5, 81.5, 1]
    ], dtype=np.float64)

    # 投影计算
    uv = HH0 @ xy_add.T
    uv = np.round(uv[:2] / uv[2]).astype(int)

    # 有效投影点筛选
    cond = (uv[0, :] > 4) & (uv[1, :] > 4) & (uv[0, :] < w - 4) & (uv[1, :] < h - 4)
    set0 = np.where(cond)[0]
    num1 = np.sum(set0 < 3)

    if len(set0) == 0:
        return xy1, uv1, jilu

    uv_valid = uv[:, set0]

    # 核心修复：sp线性索引（列优先，乘以h）
    sp = (uv_valid[1, :] - 1) * h + uv_valid[0, :]

    # ismember逻辑
    xcsp_set = set(xcsp)
    xcsp_dict = {val: idx for idx, val in enumerate(xcsp)}
    sp_window = sp[np.newaxis, :] + spij7[:, np.newaxis]

    M1 = np.isin(sp_window, xcsp)
    M2 = np.full_like(sp_window, -1, dtype=int)

    # 逐列赋值
    for col in range(M1.shape[1]):
        match_rows = np.where(M1[:, col])[0]
        if len(match_rows) > 0:
            row = match_rows[0]
            val = sp_window[row, col]
            if val in xcsp_dict:
                M2[row, col] = xcsp_dict[val]

    # 每列最大值和索引
    m = M1.max(axis=0)
    index0 = M1.argmax(axis=0)
    sk = np.where(m)[0]

    # sk1/sk2筛选
    sk1 = sk[sk < num1]
    sk2 = sk[sk >= num1]

    # ====================== 辅助函数：安全计算S值（带边界检查） ======================
    def safe_compute_S(base_idx, spij_arr):
        """安全计算S值，只取有效索引"""
        indices = base_idx + spij_arr
        # 检查3层：base, base+mm, base+2mm
        valid_mask = (indices >= 0) & (indices < max_index) & \
                     (indices + mm >= 0) & (indices + mm < max_index) & \
                     (indices + 2 * mm >= 0) & (indices + 2 * mm < max_index)
        valid_indices = indices[valid_mask]
        if len(valid_indices) == 0:
            return 0.0  # 无有效索引，返回0
        # 只对有效索引求和
        return np.sum(Im0[valid_indices] + Im0[valid_indices + mm] + Im0[valid_indices + 2 * mm])

    # ====================== sk1 处理 ======================
    if len(sk1) > 0:
        sk1_idx = sk1[0]
        orig_idx = set0[sk1_idx]
        row_idx = index0[sk1_idx]
        saddle_idx = M2[row_idx, sk1_idx]

        if saddle_idx >= 0:
            uv1 = np.vstack([uv1, np.append(xc[saddle_idx], 1.0)])
            min0 = 80.5 + orig_idx * 15
            new_part = xy1[:, :2] + np.array([16.5, min0])
            new_part = np.column_stack([new_part, np.ones(xy1.shape[0])])
            xy1 = np.vstack([new_part, [24.0, 29.0, 1.0]])

            xy0 = np.array([7.5, 0, 1])
            uv0 = HH0 @ xy0
            uv0 = uv0[:2] / uv0[2]
            uv0 = uv0 - xc[saddle_idx]
            uv0 = uv0 / np.linalg.norm(uv0)

            v1 = np.array([corner['v1'][saddle_idx, 0], -corner['v1'][saddle_idx, 1]])
            abc1 = np.concatenate([v1, [-np.dot(v1, xc[saddle_idx])]])
            v2 = np.array([corner['v2'][saddle_idx, 0], -corner['v2'][saddle_idx, 1]])
            abc2 = np.concatenate([v2, [-np.dot(v2, xc[saddle_idx])]])

            a1b1 = np.array([np.dot(abc1, HH0[:, 0]), np.dot(abc1, HH0[:, 1])])
            a1b1 = a1b1 / np.linalg.norm(a1b1)
            a2b2 = np.array([np.dot(abc2, HH0[:, 0]), np.dot(abc2, HH0[:, 1])])
            a2b2 = a2b2 / np.linalg.norm(a2b2)
            max1 = max(np.abs(a1b1[0]), np.abs(a2b2[0]))

            # 核心修复：安全计算S值
            base_val = xcsp[saddle_idx]
            numm1 = base_val - int(round(7 * uv0[0])) * h + int(round(7 * uv0[1]))
            numm2 = base_val + int(round(7 * uv0[1])) * h + int(round(7 * uv0[0]))
            numm3 = base_val + int(round(7 * (uv0[1] + uv0[0]))) * h + int(round(7 * (-uv0[1] + uv0[0])))
            numm4 = base_val + int(round(7 * (uv0[1] - uv0[0]))) * h + int(round(7 * (uv0[1] + uv0[0])))

            S1 = safe_compute_S(numm1, spij3)
            S2 = safe_compute_S(numm2, spij3)
            S3 = safe_compute_S(numm3, spij3)
            S4 = safe_compute_S(numm4, spij3)

            if max1 > 0.97:
                if S3 < S4:
                    jilu.append([2, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                    xy1[:, 0] += 162
                else:
                    jilu.append([1, xc[saddle_idx, 0], xc[saddle_idx, 1]])
            else:
                if S1 < S2:
                    jilu.append([3, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                    xy1[:, 1] = 296 - xy1[:, 1]
                    xy1[:, 0] = 210 - xy1[:, 0]
                else:
                    jilu.append([4, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                    xy1[:, 1] = 296 - xy1[:, 1]
                    xy1[:, 0] = 48 - xy1[:, 0]

    # ====================== sk2 处理 ======================
    if len(sk2) > 0:
        sk2_idx = sk2[0]
        orig_idx = set0[sk2_idx]
        row_idx = index0[sk2_idx]
        saddle_idx = M2[row_idx, sk2_idx]

        if saddle_idx >= 0:
            if len(sk1) > 0:
                HH1 = compute_homography(xy1[:, :2], uv1[:, :2])
                uv1 = np.vstack([uv1, np.append(xc[saddle_idx], 1.0)])

                v1 = np.array([corner['v1'][saddle_idx, 0], -corner['v1'][saddle_idx, 1]])
                abc1 = np.concatenate([v1, [-np.dot(v1, xc[saddle_idx])]])
                v2 = np.array([corner['v2'][saddle_idx, 0], -corner['v2'][saddle_idx, 1]])
                abc2 = np.concatenate([v2, [-np.dot(v2, xc[saddle_idx])]])

                a1b1 = np.array([np.dot(abc1, HH1[:, 0]), np.dot(abc1, HH1[:, 1])])
                a1b1 = a1b1 / np.linalg.norm(a1b1)
                a2b2 = np.array([np.dot(abc2, HH1[:, 0]), np.dot(abc2, HH1[:, 1])])
                a2b2 = a2b2 / np.linalg.norm(a2b2)
                max1 = max(np.abs(a1b1[0]), np.abs(a2b2[0]))

                xy0 = np.array([(np.max(xy1[:, 0]) + np.min(xy1[:, 0])) / 2, np.max(xy1[:, 1]), 1])
                uv0 = HH1 @ xy0
                uv0 = uv0[:2] / uv0[2]
                uv0 = uv0 - xc[saddle_idx]
                uv0 = uv0 / np.linalg.norm(uv0)
                xy1 = np.vstack([xy1, [24.0, 267.0, 1.0]])

                # 安全计算S值
                base_val = xcsp[saddle_idx]
                numm1 = base_val - int(round(7 * uv0[0])) * h + int(round(7 * uv0[1]))
                numm2 = base_val + int(round(7 * uv0[1])) * h + int(round(7 * uv0[0]))
                numm3 = base_val + int(round(7 * (uv0[1] + uv0[0]))) * h + int(round(7 * (-uv0[1] + uv0[0])))
                numm4 = base_val + int(round(7 * (uv0[1] - uv0[0]))) * h + int(round(7 * (uv0[1] + uv0[0])))

                S1 = safe_compute_S(numm1, spij3)
                S2 = safe_compute_S(numm2, spij3)
                S3 = safe_compute_S(numm3, spij3)
                S4 = safe_compute_S(numm4, spij3)

                if max1 > 0.97:
                    if S3 < S4:
                        jilu.append([2, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                        xy1[-1, :] = [186.0, 29.0, 1.0]
                    else:
                        jilu.append([1, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                        xy1[-1, :] = [24.0, 29.0, 1.0]
                else:
                    if S1 < S2:
                        jilu.append([3, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                        xy1[-1, :] = [186.0, 267.0, 1.0]
                    else:
                        jilu.append([4, xc[saddle_idx, 0], xc[saddle_idx, 1]])
            else:
                xy0 = np.array([7.5, np.max(xy1[:, 1]), 1])
                HH1 = compute_homography(xy1[:, :2], uv1[:, :2])
                uv1 = np.vstack([uv1, np.append(xc[saddle_idx], 1.0)])
                min0 = 215.5 - (13 - orig_idx) * 15
                new_part = xy1[:, :2] + np.array([16.5, min0])
                new_part = np.column_stack([new_part, np.ones(xy1.shape[0])])
                xy1 = np.vstack([new_part, [24.0, 267.0, 1.0]])

                uv0 = HH1 @ xy0
                uv0 = uv0[:2] / uv0[2]
                uv0 = uv0 - xc[saddle_idx]
                uv0 = uv0 / np.linalg.norm(uv0)

                v1 = np.array([corner['v1'][saddle_idx, 0], -corner['v1'][saddle_idx, 1]])
                abc1 = np.concatenate([v1, [-np.dot(v1, xc[saddle_idx])]])
                v2 = np.array([corner['v2'][saddle_idx, 0], -corner['v2'][saddle_idx, 1]])
                abc2 = np.concatenate([v2, [-np.dot(v2, xc[saddle_idx])]])

                a1b1 = np.array([np.dot(abc1, HH1[:, 0]), np.dot(abc1, HH1[:, 1])])
                a1b1 = a1b1 / np.linalg.norm(a1b1)
                a2b2 = np.array([np.dot(abc2, HH1[:, 0]), np.dot(abc2, HH1[:, 1])])
                a2b2 = a2b2 / np.linalg.norm(a2b2)
                max1 = max(np.abs(a1b1[0]), np.abs(a2b2[0]))

                # 安全计算S值
                base_val = xcsp[saddle_idx]
                numm1 = base_val - int(round(7 * uv0[0])) * h + int(round(7 * uv0[1]))
                numm2 = base_val + int(round(7 * uv0[1])) * h + int(round(7 * uv0[0]))
                numm3 = base_val + int(round(7 * (uv0[1] + uv0[0]))) * h + int(round(7 * (-uv0[1] + uv0[0])))
                numm4 = base_val + int(round(7 * (uv0[1] - uv0[0]))) * h + int(round(7 * (uv0[1] + uv0[0])))

                S1 = safe_compute_S(numm1, spij3)
                S2 = safe_compute_S(numm2, spij3)
                S3 = safe_compute_S(numm3, spij3)
                S4 = safe_compute_S(numm4, spij3)

                if max1 > 0.97:
                    if S3 < S4:
                        jilu.append([2, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                        xy1[:, 1] = 296 - xy1[:, 1]
                        xy1[:, 0] += 162
                    else:
                        jilu.append([1, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                        xy1[:, 1] = 296 - xy1[:, 1]
                else:
                    if S1 < S2:
                        jilu.append([3, xc[saddle_idx, 0], xc[saddle_idx, 1]])
                        xy1[:, 0] += 162
                    else:
                        jilu.append([4, xc[saddle_idx, 0], xc[saddle_idx, 1]])
    return xy1, uv1, jilu

# ======================== 主程序（核心修改：不做错误归一化） ========================
if __name__ == '__main__':
    # 请修改为你的图像路径
    img_path = r'C:\Users\30772\Documents\2022nov1112\hewei005.bmp'
    print("读取图像...")
    img_gray = imread_gray(img_path)
    img_gray = gaussian_filter(img_gray, sigma=0.5)

    # 模糊评估
    blur_var = measure_blur_level(img_gray)
    print(f"拉普拉斯方差: {blur_var:.2f}")
    tau_nms = 0.02 if blur_var < 50 else 0.025
    tau_score = 0.03 if blur_var < 50 else 0.04
    print(f"NMS阈值: {tau_nms:.3f}, 评分阈值: {tau_score:.3f}")

    h, w = img_gray.shape
    nn = [h, w]
    mm = h

    # spij3/spij7列优先生成
    jj3, ii3 = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing='xy')
    spij3 = (ii3.ravel(order='F') * h + jj3.ravel(order='F')).astype(int)

    jj7, ii7 = np.meshgrid(np.arange(-7, 8), np.arange(-7, 8), indexing='xy')
    spij7 = (ii7.ravel(order='F') * h + jj7.ravel(order='F')).astype(int)

    # Im0列优先展开
    Im0 = img_gray.ravel(order='C')

    t0 = time.time()
    print("计算梯度...")
    angle_map, mag_map, img_du, img_dv = compute_gradients(img_gray)
    img_norm = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)

    print("多尺度模板滤波...")
    radius_list = [4, 8, 12]
    corner_map = filter_image_with_templates_fft(img_norm, radius_list)

    print("非极大值抑制...")
    corners_init = non_maximum_suppression_fast(corner_map, n=3, tau=tau_nms, margin=5)
    print(f"初始候选点: {len(corners_init)}")

    if len(corners_init) == 0:
        print("未检测到候选点")
        exit()

    MAX_CANDIDATES = 3000
    if len(corners_init) > MAX_CANDIDATES:
        resp_vals = corner_map[corners_init[:,1].astype(int), corners_init[:,0].astype(int)]
        idx = np.argsort(resp_vals)[-MAX_CANDIDATES:]
        corners_init = corners_init[idx]
        print(f"裁剪至 {MAX_CANDIDATES} 个候选点")

    print("亚像素精化...")
    refined_p, refined_v1, refined_v2 = refine_corners_vectorized(
        img_du, img_dv, angle_map, mag_map, corners_init, r=10, img_gray=img_norm)
    print(f"精化后保留: {len(refined_p)} 个")

    if len(refined_p) == 0:
        print("精化后无有效点")
        exit()

    print("计算评分...")
    scores = score_corners_batch(img_norm, mag_map, refined_p, refined_v1, refined_v2, radius_list)
    keep = scores >= tau_score
    final_points = refined_p[keep]
    final_v1 = refined_v1[keep]
    final_v2 = refined_v2[keep]
    final_scores = scores[keep]
    print(f"评分过滤后: {len(final_points)} 个")

    if len(final_points) > 0:
        dirs_array = np.stack([final_v1, final_v2], axis=1)
        final_points, dirs_array = merge_close_points(final_points, dirs_array, final_scores,
                                                      dist_thresh=8.0, ratio=0.5)
        final_v1 = dirs_array[:, 0]
        final_v2 = dirs_array[:, 1]
        print(f"合并后: {len(final_points)} 个")
        final_v1, final_v2 = adjust_orientation_handedness(final_v1, final_v2)

    print(f"鞍点检测耗时: {time.time()-t0:.2f} 秒")
    print("\n开始棋盘格生长...")

    points = final_points.copy()
    directions = np.stack([final_v1, final_v2], axis=1)
    found_chessboards = []
    sorted_idx = np.argsort(final_scores)[::-1]

    for start_idx in sorted_idx[:min(200, len(points))]:
        quad_pts, quad_idx = get_little_four_P_py(points, directions, start_idx, img_gray.shape)
        if quad_pts is not None:
            img_pts, world_pts, H, used_idx = grow_chessboard_region(points, directions, quad_idx)
            # 规范化：平移到原点 + 长边对齐（同步调整世界坐标和图像坐标）
            world_pts = world_pts - np.min(world_pts, axis=0)
            max_xy = np.max(world_pts, axis=0)
            if max_xy[0] > max_xy[1]:
                # 交换列
                world_pts = world_pts[:, [1, 0]]
                world_pts[:, 0] = 15.0 - world_pts[:, 0]
                # 重要：图像坐标也要同步交换顺序（但图像坐标本身不变，只需保持与world_pts的对应）
                # 由于world_pts只是数值变换，不改变点的对应关系，所以img_pts无需修改
                # 只需重新计算H
            H = compute_homography(world_pts, img_pts)
            if len(img_pts) > 10:
                is_new = True
                for exist in found_chessboards:
                    overlap = len(used_idx & exist[3])
                    if overlap > 0.5 * len(used_idx) or overlap > 0.5 * len(exist[3]):
                        is_new = False
                        if len(img_pts) > len(exist[0]):
                            exist = (img_pts, world_pts, H, used_idx)
                        break
                if is_new:
                    found_chessboards.append((img_pts, world_pts, H, used_idx))
                    unique_x = np.unique(np.round(world_pts[:, 0], decimals=1))
                    unique_y = np.unique(np.round(world_pts[:, 1], decimals=1))
                    print(f"发现新棋盘格，角点数: {len(img_pts)}，列数: {len(unique_x)}，行数: {len(unique_y)}")
                if len(found_chessboards) >= 10:
                    break

    # 筛选两列棋盘格
    two_col_candidates = []
    for chess in found_chessboards:
        _, world_pts, H, _ = chess
        unique_x = np.unique(np.round(world_pts[:, 0], decimals=1))
        unique_y = np.unique(np.round(world_pts[:, 1], decimals=1))
        if min(len(unique_x), len(unique_y)) == 2 and max(len(unique_x), len(unique_y)) >= 3:
            two_col_candidates.append(chess)
    two_col_candidates.sort(key=lambda x: len(x[0]), reverse=True)
    print(f"找到 {len(two_col_candidates)} 个两列棋盘格候选")

    # xcsp计算
    x = final_points[:, 0]
    y = final_points[:, 1]
    xcsp = (np.round(y).astype(int)) * h + (np.round(x).astype(int) + 1)

    best_left = None
    best_right = None
    if len(two_col_candidates) >= 2:
        best_left = two_col_candidates[0]
        best_right = two_col_candidates[1]
    elif len(two_col_candidates) == 1:
        best_left = two_col_candidates[0]
    else:
        print("未检测到足够的两列棋盘格")

    if best_left is not None or best_right is not None:
        corner = {'v1': final_v1, 'v2': final_v2}
        jilu = []

        # 处理左棋盘格（核心修改：直接使用原始world_pts，不做归一化）
        if best_left is not None:
            left_img_pts = best_left[0].copy()
            left_world_pts = best_left[1].copy()

            # 简单平移到原点，不做翻转
            left_world_pts = left_world_pts - np.min(left_world_pts, axis=0)

            # 确保长边为y轴（和MATLAB一致）
            max_xy = np.max(left_world_pts, axis=0)
            if max_xy[0] > max_xy[1]:
                left_world_pts = left_world_pts[:, [1, 0]]
                left_world_pts[:, 0] = 15.0 - left_world_pts[:, 0]

            left_H = compute_homography(left_world_pts, left_img_pts)
            xy1_left = np.column_stack([left_world_pts, np.ones(len(left_world_pts))])
            uv1_left = np.column_stack([left_img_pts, np.ones(len(left_img_pts))])

            xy1_left, uv1_left, jilu_left = get_mark_cord(
                final_points, corner, xy1_left, uv1_left, left_H,
                nn, spij3, spij7, [], xcsp, Im0, mm
            )
            jilu.extend(jilu_left)
            print(f"左棋盘格识别到 {len(jilu_left)} 个标记点")

        # 处理右棋盘格
        if best_right is not None:
            right_img_pts = best_right[0].copy()
            right_world_pts = best_right[1].copy()

            # 简单平移到原点
            right_world_pts = right_world_pts - np.min(right_world_pts, axis=0)

            max_xy = np.max(right_world_pts, axis=0)
            if max_xy[0] > max_xy[1]:
                right_world_pts = right_world_pts[:, [1, 0]]
                right_world_pts[:, 0] = 15.0 - right_world_pts[:, 0]

            right_H = compute_homography(right_world_pts, right_img_pts)
            xy1_right = np.column_stack([right_world_pts, np.ones(len(right_world_pts))])
            uv1_right = np.column_stack([right_img_pts, np.ones(len(right_img_pts))])

            xy1_right, uv1_right, jilu_right = get_mark_cord(
                final_points, corner, xy1_right, uv1_right, right_H,
                nn, spij3, spij7, [], xcsp, Im0, mm
            )
            jilu.extend(jilu_right)
            print(f"右棋盘格识别到 {len(jilu_right)} 个标记点")

        # 输出jilu信息
        print("\n识别到的标记点 (jilu)：")
        for rec in jilu:
            print(f"  类型 {rec[0]}，坐标 ({rec[1]:.1f}, {rec[2]:.1f})")

        # 可视化
        plt.figure(figsize=(12, 10))
        plt.imshow(img_gray, cmap='gray')
        plt.plot(points[:, 0], points[:, 1], 'y.', markersize=1, alpha=0.3)
        if best_left is not None:
            plt.plot(left_img_pts[:, 0], left_img_pts[:, 1], 'b+', markersize=4, label='Left chess')
        if best_right is not None:
            plt.plot(right_img_pts[:, 0], right_img_pts[:, 1], 'g+', markersize=4, label='Right chess')
        colors = {1: 'r', 2: 'm', 3: 'c', 4: 'orange'}
        for rec in jilu:
            plt.plot(rec[1], rec[2], 'o', color=colors.get(rec[0], 'w'),
                     markersize=8, markeredgecolor='k')
            plt.text(rec[1] + 5, rec[2] - 5, str(rec[0]), color='white', fontsize=10, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, pad=1))
        plt.legend()
        plt.title(f'Chessboards and Marks (jilu count: {len(jilu)})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("未检测到任何两列棋盘格，仅显示鞍点结果。")
        plt.figure(figsize=(12, 10))
        plt.imshow(img_gray, cmap='gray')
        plt.plot(points[:, 0], points[:, 1], 'y.', markersize=1, alpha=0.3)
        plt.title('Saddle points only')
        plt.axis('off')
        plt.tight_layout()
        plt.show()