from __future__ import annotations

import ast
import json
import math
import re
import runpy
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "data"
OUT_FILE = DATA_DIR / "course-data.json"
RADIUS = 10.0
MAX_PATH_POINTS = 90


COURSE_COLORS = {
    "gaoshu-shang": "#2f7dd1",
    "gaoshu-xia": "#d85b57",
    "linear-algebra": "#2e9b72",
    "probability-statistics": "#a36a2d",
}

CHAPTER_TITLES = {
    "gaoshu-shang": {
        "1": "第一章 函数与极限",
        "2": "第二章 导数与微分",
        "3": "第三章 微分中值定理与导数的应用",
        "4": "第四章 不定积分",
        "5": "第五章 定积分",
        "6": "第六章 定积分的应用",
        "7": "第七章 常微分方程",
    },
    "gaoshu-xia": {
        "8": "第八章 空间解析几何",
        "9": "第九章 多元函数微分法及其应用",
        "10": "第十章 重积分",
        "11": "第十一章 曲线积分与曲面积分",
        "12": "第十二章 无穷级数",
    },
    "linear-algebra": {
        "0": "新手村",
        "1": "第一章 行列式",
        "2": "第二章 矩阵及其运算",
        "3": "第三章 矩阵的初等变换与线性方程组",
        "4": "第四章 向量组的线性相关性",
        "5": "第五章 相似矩阵及二次型",
    },
    "probability-statistics": {
        "1": "第一章 概率论的基本概念",
        "2": "第二章 随机变量及其分布",
        "3": "第三章 多维随机变量及其分布",
        "4": "第四章 随机变量的数字特征",
        "5": "第五章 大数定律和中心极限定理",
        "6": "第六章 统计量与抽样分布",
        "7": "第七章 参数估计",
        "8": "第八章 假设检验",
    },
}


def find_source(prefix: str) -> Path:
    for path in DATA_DIR.glob("*.py"):
        if path.name.startswith(prefix):
            return path
    raise FileNotFoundError(prefix)


def normalize_xyz(xyz) -> np.ndarray:
    arr = np.asarray(xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.array([0.0, 0.0, RADIUS], dtype=float)
    return arr / norm * RADIUS


def coord_obj(xyz) -> dict:
    x, y, z = normalize_xyz(xyz)
    # Source scripts use z as vertical. Three.js uses y as vertical, so phi/psi
    # are computed in the same coordinate system used by the renderer.
    tx, ty, tz = x, z, y
    phi = math.atan2(tz, tx)
    psi = math.asin(max(-1.0, min(1.0, ty / RADIUS)))
    return {
        "x": round(float(x), 5),
        "y": round(float(y), 5),
        "z": round(float(z), 5),
        "phi": round(phi, 6),
        "psi": round(psi, 6),
    }


def curve_array_to_path(curve) -> list[dict]:
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] != 3:
        return []
    return [coord_obj(row) for row in arr]


def merge_curve_segments(segments) -> list[dict]:
    path: list[dict] = []
    for segment in segments:
        points = curve_array_to_path(segment)
        if not points:
            continue
        if path and points:
            first = points[0]
            last = path[-1]
            if distance_obj(first, last) < 1e-4:
                points = points[1:]
        path.extend(points)
    return downsample_path(path, MAX_PATH_POINTS)


def downsample_path(path: list[dict], max_points: int) -> list[dict]:
    if len(path) <= max_points:
        return path
    indices = np.linspace(0, len(path) - 1, max_points, dtype=int)
    return [path[int(i)] for i in indices]


def distance_obj(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def source_vec(point: dict) -> np.ndarray:
    return np.array([point["x"], point["y"], point["z"]], dtype=float)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = normalize_xyz(a) / RADIUS
    b = normalize_xyz(b) / RADIUS
    dot = float(np.clip(np.dot(a, b), -0.9995, 0.9995))
    omega = math.acos(dot)
    if omega < 1e-5:
        return normalize_xyz((1 - t) * a + t * b)
    so = math.sin(omega)
    return normalize_xyz((math.sin((1 - t) * omega) / so) * a + (math.sin(t * omega) / so) * b)


def spherical_hull(paths: list[list[dict]], bins: int = 56) -> list[dict]:
    vectors = [source_vec(point) for path in paths for point in path]
    vectors = [v for v in vectors if np.linalg.norm(v) > 1e-8]
    if len(vectors) < 3:
        return []
    center = normalize_xyz(np.mean(vectors, axis=0)) / RADIUS
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(center, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, center)
    u = u / np.linalg.norm(u)
    v = np.cross(center, u)
    buckets: dict[int, tuple[float, np.ndarray]] = {}
    for item in vectors:
        p = normalize_xyz(item) / RADIUS
        tangent = p - center * np.dot(p, center)
        radial = float(np.linalg.norm(tangent))
        if radial < 1e-6:
            continue
        angle = math.atan2(float(np.dot(tangent, v)), float(np.dot(tangent, u)))
        bucket = int(((angle + math.pi) / (2 * math.pi)) * bins) % bins
        old = buckets.get(bucket)
        if old is None or radial > old[0]:
            buckets[bucket] = (radial, p)
    if len(buckets) < 3:
        return []
    ordered = [buckets[key][1] * RADIUS for key in sorted(buckets)]
    return [coord_obj(item) for item in ordered]


def chapter_key(title: str) -> str:
    if title.startswith("新手村"):
        return "0"
    match = re.match(r"(\d+)", title)
    if match:
        value = match.group(1)
        if value == "0":
            return "0"
        return value
    return "0"


def point_label(label: str, xyz, point_id: str) -> dict:
    item = coord_obj(xyz)
    item.update({"id": point_id, "label": label})
    return item


def knowledge_for_path(course_id: str, section_id: str, path: list[dict], count: int = 3) -> list[dict]:
    if len(path) < 3:
        return []
    vectors = [source_vec(p) for p in path]
    center = normalize_xyz(np.mean(vectors, axis=0))
    items = []
    for idx in range(count):
        anchor = vectors[int((idx + 0.5) * len(vectors) / count) % len(vectors)]
        t = 0.48 + 0.12 * (idx % 2)
        point = coord_obj(slerp(center, anchor, t))
        point.update({"id": f"{course_id}-{section_id}-k{idx + 1}", "label": f"K{idx + 1}"})
        items.append(point)
    return items


def build_course(course_id: str, title: str, sections: list[dict], labels: dict[str, object]) -> dict:
    chapter_titles = CHAPTER_TITLES[course_id]
    for index, section in enumerate(sections):
        key = chapter_key(section["title"])
        section["id"] = f"{course_id}-section-{index + 1}"
        section["chapterKey"] = key
        section["chapter"] = chapter_titles.get(key, f"第{key}章")
        section["knowledge"] = knowledge_for_path(course_id, section["id"], section["path"])

    chapter_groups: dict[str, list[dict]] = defaultdict(list)
    for section in sections:
        chapter_groups[section["chapterKey"]].append(section)

    chapters = []
    for key, grouped_sections in chapter_groups.items():
        chapters.append(
            {
                "id": f"{course_id}-chapter-{key}",
                "key": key,
                "title": chapter_titles.get(key, f"第{key}章"),
                "path": spherical_hull([section["path"] for section in grouped_sections]),
                "sectionIds": [section["id"] for section in grouped_sections],
            }
        )

    points = [
        point_label(label, xyz, f"{course_id}-p{idx + 1}")
        for idx, (label, xyz) in enumerate(labels.items())
    ]

    return {
        "id": course_id,
        "title": title,
        "color": COURSE_COLORS[course_id],
        "points": points[:72],
        "chapters": chapters,
        "sections": sections,
    }


def resolve_gaoshu_up_instruction(ns: dict, instruction) -> np.ndarray:
    kind = instruction[0]
    if kind == "arc":
        _, r, t_start, t_end, z_plane = instruction
        return np.column_stack(ns["generate_arc"](r, t_start, t_end, z_plane, ns["R"]))
    if kind == "arc_rev":
        _, r, t_start, t_end, z_plane = instruction
        return np.column_stack(ns["generate_arc"](r, t_end, t_start, z_plane, ns["R"]))
    if kind == "line":
        _, p1_name, p2_name = instruction
        return np.column_stack(ns["generate_line"](ns["all_points"][p1_name], ns["all_points"][p2_name], ns["R"]))
    return np.empty((0, 3))


def load_gaoshu_up() -> dict:
    path = find_source("\u9ad8\u7b49\u6570\u5b66\u4e0a")
    ns = runpy.run_path(str(path))
    sections = []
    label_names = set()
    for name, instructions in ns["sections"].items():
        segments = [resolve_gaoshu_up_instruction(ns, item) for item in instructions]
        sections.append({"title": name, "path": merge_curve_segments(segments)})
        for item in instructions:
            if item[0] == "line":
                label_names.add(item[1])
                label_names.add(item[2])
    labels = {name: ns["all_points"][name] for name in sorted(label_names)}
    plt.close("all")
    return build_course("gaoshu-shang", "高等数学上", sections, labels)


def load_gaoshu_down() -> dict:
    path = find_source("\u9ad8\u7b49\u6570\u5b66\u4e0b")
    ns = runpy.run_path(str(path))
    sections = [
        {"title": name, "path": downsample_path(curve_array_to_path(curve), MAX_PATH_POINTS)}
        for name, curve in ns["chapter_curves"].items()
    ]
    labels = ns["collect_all_label_points"]()
    plt.close("all")
    return build_course("gaoshu-xia", "高等数学下", sections, labels)


def load_linear_algebra() -> dict:
    path = find_source("\u7ebf\u6027\u4ee3\u6570")
    ns = runpy.run_path(str(path))
    arcs = ns["create_all_arcs"]()
    regions = ns["create_all_regions"](arcs)
    sections = [
        {"title": name, "path": merge_curve_segments(curves)}
        for name, curves in regions.items()
    ]
    labels = ns["collect_all_label_points"](arcs)
    plt.close("all")
    return build_course("linear-algebra", "线性代数", sections, labels)


class RecordingAxes:
    def __init__(self) -> None:
        self.records: list[tuple[str, np.ndarray]] = []

    def plot(self, x, y, z, color=None, **_kwargs):
        if color is None:
            return
        arr = np.column_stack([x, y, z])
        self.records.append((str(color).upper(), arr))

    def plot_surface(self, *_args, **_kwargs):
        return


CN_NUM = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def cn_number(text: str) -> int:
    if text == "十":
        return 10
    if text.startswith("十"):
        return 10 + CN_NUM.get(text[1:], 0)
    if text.endswith("十"):
        return CN_NUM.get(text[:-1], 1) * 10
    if "十" in text:
        left, right = text.split("十", 1)
        return CN_NUM.get(left, 1) * 10 + CN_NUM.get(right, 0)
    return CN_NUM.get(text, 0)


def probability_color_titles(source: str) -> dict[str, str]:
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def draw_all_sections"))
    color_titles: dict[str, str] = {}
    chapter = 0
    section = 0
    current_title = ""
    for line in lines[start:]:
        chapter_match = re.search(r"#\s*第([一二三四五六七八九十]+)章[：:](.+)", line)
        if chapter_match:
            chapter = cn_number(chapter_match.group(1))
            section = 0
            continue
        section_match = re.search(r"#\s*第([一二三四五六七八九十]+)节[：:]?\s*(.+)?", line)
        if section_match:
            section = cn_number(section_match.group(1)) or section + 1
            current_title = (section_match.group(2) or "").strip()
            continue
        color_match = re.search(r"color=['\"](#[0-9A-Fa-f]{6})['\"]", line)
        if color_match and chapter and section and current_title:
            color = color_match.group(1).upper()
            color_titles.setdefault(color, f"{chapter}.{section} {current_title}")
    return color_titles


def load_probability() -> dict:
    path = find_source("\u6982\u7387\u8bba")
    ns = runpy.run_path(str(path))
    source = path.read_text(encoding="utf-8")
    color_titles = probability_color_titles(source)
    axes = RecordingAxes()
    ns["draw_all_sections"](axes)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for color, arr in axes.records:
        grouped[color].append(arr)

    sections = []
    for color, title in color_titles.items():
        curves = grouped.get(color, [])
        if curves:
            sections.append({"title": title, "path": merge_curve_segments(curves)})

    labels = {label: point for point, label in ns.get("point_label_list", [])}
    plt.close("all")
    return build_course("probability-statistics", "概率论与数理统计", sections, labels)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    data = {
        "schema": 2,
        "radius": RADIUS,
        "generatedFrom": [
            "高等数学上球面各变量名.py",
            "高等数学下各段变量名.py",
            "线性代数各段变量名.py",
            "概率论与数理统计各变量名.py",
        ],
        "courses": [
            load_gaoshu_up(),
            load_gaoshu_down(),
            load_linear_algebra(),
            load_probability(),
        ],
    }
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for course in data["courses"]:
        print(
            f"{course['title']}: "
            f"{len(course['points'])} points, "
            f"{len(course['chapters'])} chapters, "
            f"{len(course['sections'])} sections"
        )
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
