import * as THREE from 'three';

const TAU = Math.PI * 2;
const FIELD_WIDTH = 768;
const FIELD_HEIGHT = 384;

const CHAPTER_PALETTE = [
  '#ef3f4f', '#16a3d6', '#55b73f', '#ff9f1c',
  '#8c52d6', '#eb5e28', '#00a896', '#3f6ad8',
  '#db4fa4', '#e3b341', '#2d9c5a', '#6b5bd2'
];

const COURSE_CHAPTER_PALETTES = {
  // 与线性代数原始区域图的章节配色保持一致，避免第三章和第五章都呈橙色。
  'linear-algebra': ['#ff4444', '#00cc99', '#3366ff', '#ff9933', '#9966ff', '#ff66cc']
};

const fibonacciCandidates = Array.from({ length: 640 }, (_, index) => {
  const offset = 2 / 640;
  const y = index * offset - 1 + offset / 2;
  const radius = Math.sqrt(Math.max(0, 1 - y * y));
  const angle = index * Math.PI * (3 - Math.sqrt(5));
  return new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius);
});

export function smoothstep(edge0, edge1, value) {
  const t = THREE.MathUtils.clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

export function sourceToVector(point, radius = 10) {
  if (Number.isFinite(point?.x) && Number.isFinite(point?.y) && Number.isFinite(point?.z)) {
    const vector = new THREE.Vector3(point.x, point.z, point.y);
    if (vector.lengthSq() > 0.0001) return vector.normalize().multiplyScalar(radius);
  }
  const phi = Number(point?.phi) || 0;
  const psi = Number(point?.psi) || 0;
  const cp = Math.cos(psi);
  return new THREE.Vector3(
    radius * cp * Math.cos(phi),
    radius * Math.sin(psi),
    radius * cp * Math.sin(phi)
  );
}

export function vectorToSource(localVector, target, radius = 10) {
  const vector = localVector.clone().normalize().multiplyScalar(radius);
  target.x = Number(vector.x.toFixed(5));
  target.y = Number(vector.z.toFixed(5));
  target.z = Number(vector.y.toFixed(5));
  target.phi = Number(Math.atan2(vector.z, vector.x).toFixed(6));
  target.psi = Number(Math.asin(THREE.MathUtils.clamp(vector.y / radius, -1, 1)).toFixed(6));
}

export function slerpUnit(a, b, t) {
  const start = a.clone().normalize();
  const end = b.clone().normalize();
  const dot = THREE.MathUtils.clamp(start.dot(end), -0.9995, 0.9995);
  const omega = Math.acos(dot);
  if (omega < 0.0001) return start.lerp(end, t).normalize();
  const sinOmega = Math.sin(omega);
  return start
    .multiplyScalar(Math.sin((1 - t) * omega) / sinOmega)
    .add(end.multiplyScalar(Math.sin(t * omega) / sinOmega))
    .normalize();
}

function lighten(color, amount) {
  return new THREE.Color(color).lerp(new THREE.Color(0xffffff), amount);
}

function sectionColor(baseColor, index, total) {
  const source = new THREE.Color(baseColor);
  const hsl = {};
  source.getHSL(hsl);
  const phase = total <= 1 ? 0.5 : index / (total - 1);
  const lightness = [0.46, 0.66, 0.54, 0.73, 0.39, 0.61, 0.50, 0.69][index % 8];
  const saturation = [0.92, 0.76, 0.88, 0.80, 0.90, 0.82, 0.70, 0.86][index % 8];
  return new THREE.Color().setHSL(
    (hsl.h + (phase - 0.5) * 0.045 + 1) % 1,
    THREE.MathUtils.clamp(saturation, 0.62, 0.96),
    THREE.MathUtils.clamp(lightness, 0.34, 0.76)
  );
}

export function prepareCourse(course) {
  course.__vertexMap = new Map(course.vertices.map((vertex) => [vertex.id, vertex]));
  const chapterPalette = COURSE_CHAPTER_PALETTES[course.id] || CHAPTER_PALETTE;
  course.chapters.forEach((chapter, chapterIndex) => {
    chapter.__color = new THREE.Color(chapterPalette[chapterIndex % chapterPalette.length]);
  });
  course.sections.forEach((section) => {
    const chapterIndex = Math.max(0, course.chapters.findIndex((chapter) => chapter.id === section.chapterId));
    const siblings = course.sections.filter((item) => item.chapterId === section.chapterId);
    const sectionIndex = Math.max(0, siblings.findIndex((item) => item.id === section.id));
    section.__color = sectionColor(
      course.chapters[chapterIndex]?.__color || new THREE.Color(course.color),
      sectionIndex,
      siblings.length
    );
  });
}

export function readableTextColor(color) {
  const source = color instanceof THREE.Color ? color : new THREE.Color(color);
  const luminance = 0.2126 * source.r + 0.7152 * source.g + 0.0722 * source.b;
  return luminance < 0.38 ? '#ffffff' : '#102033';
}

export function regionBoundaryVectors(course, region, radius = 1) {
  return (region.vertexIds || [])
    .map((vertexId) => course.__vertexMap.get(vertexId))
    .filter(Boolean)
    .map((vertex) => sourceToVector(vertex, radius));
}

export function sampleClosedBoundary(course, region, stepsPerEdge = 9, radius = 1) {
  const controls = regionBoundaryVectors(course, region, 1);
  const sampled = [];
  for (let index = 0; index < controls.length; index += 1) {
    const start = controls[index];
    const end = controls[(index + 1) % controls.length];
    for (let step = 0; step < stepsPerEdge; step += 1) {
      sampled.push(slerpUnit(start, end, step / stepsPerEdge).multiplyScalar(radius));
    }
  }
  return sampled;
}

function sphericalWinding(point, boundary) {
  const direction = point.clone().normalize();
  let winding = 0;
  for (let index = 0; index < boundary.length; index += 1) {
    const a = boundary[index].clone().normalize();
    const b = boundary[(index + 1) % boundary.length].clone().normalize();
    const tangentA = a.addScaledVector(direction, -a.dot(direction));
    const tangentB = b.addScaledVector(direction, -b.dot(direction));
    if (tangentA.lengthSq() < 1e-10 || tangentB.lengthSq() < 1e-10) {
      const onBoundary = a.dot(direction) > 0.999999 || b.dot(direction) > 0.999999;
      return { winding: 0, onBoundary };
    }
    tangentA.normalize();
    tangentB.normalize();
    const cross = new THREE.Vector3().crossVectors(tangentA, tangentB);
    winding += Math.atan2(direction.dot(cross), tangentA.dot(tangentB));
  }
  return { winding, onBoundary: false };
}

function boundaryInteriorSign(boundary) {
  // 用球面三角形的有向面积确定边界朝向。与“顶点平均方向”不同，
  // 该方法对凹区域同样有效，不会因为质心落到区域外而翻转内外侧。
  const origin = boundary[0];
  let signedArea = 0;
  for (let index = 1; index < boundary.length - 1; index += 1) {
    const b = boundary[index];
    const c = boundary[index + 1];
    const determinant = origin.dot(new THREE.Vector3().crossVectors(b, c));
    const denominator = 1 + origin.dot(b) + b.dot(c) + c.dot(origin);
    signedArea += 2 * Math.atan2(determinant, denominator);
  }
  return Math.sign(signedArea) || 1;
}

export function sphericalContains(point, boundary) {
  if (boundary.length < 3) return false;
  const result = sphericalWinding(point, boundary);
  if (result.onBoundary) return true;
  // 一条球面闭合曲线会把球分成两面，两面的绕数绝对值都可能接近 2π。
  // 以边界的有向面积确定内部，避免把背面的镜像区域也误判为命中。
  const interiorSign = boundaryInteriorSign(boundary);
  return result.winding * interiorSign > Math.PI;
}

function chapterRegionFromField(course, regions, direction) {
  const ids = course.__chapterRegionIds;
  if (regions !== course.chapters || ids?.length !== FIELD_WIDTH * FIELD_HEIGHT) return null;
  const local = direction.clone().normalize();
  let u = Math.atan2(local.z, -local.x) / TAU;
  if (u < 0) u += 1;
  const v = Math.asin(THREE.MathUtils.clamp(local.y, -1, 1)) / Math.PI + 0.5;
  const x = Math.min(FIELD_WIDTH - 1, Math.floor(u * FIELD_WIDTH));
  const y = Math.min(FIELD_HEIGHT - 1, Math.max(0, Math.floor((1 - v) * FIELD_HEIGHT)));
  return regions[ids[y * FIELD_WIDTH + x]] || null;
}

export function findRegionAtDirection(course, regions, direction) {
  const fieldChapter = chapterRegionFromField(course, regions, direction);
  if (fieldChapter) return fieldChapter;
  let candidates = regions;
  if (regions.some((region) => region.chapterId)) {
    const chapter = findRegionAtDirection(course, course.chapters, direction);
    const siblings = regions.filter((region) => region.chapterId === chapter?.id);
    // “新手村”没有直属小节，但与四个入门小节区域相交；空章节沿用
    // 全部小节候选，才能与小节语义图保持相同的显示和点击归属。
    if (siblings.length) candidates = siblings;
  }
  let bestRegion = null;
  let bestDot = -Infinity;
  for (let index = 0; index < candidates.length; index += 1) {
    const boundary = regionBoundaryVectors(course, candidates[index], 1);
    const site = regionLayout(course, candidates[index]).anchor;
    const siteDot = direction.dot(site);
    if (sphericalContains(direction, boundary) && siteDot > bestDot) {
      bestRegion = candidates[index];
      bestDot = siteDot;
    }
  }
  // 与语义场保持一致：数据存在细小空隙时按最近区域中心补齐归属。
  if (!bestRegion) {
    candidates.forEach((region) => {
      const site = regionLayout(course, region).anchor;
      const siteDot = direction.dot(site);
      if (siteDot > bestDot) {
        bestRegion = region;
        bestDot = siteDot;
      }
    });
  }
  return bestRegion;
}

function centroidDirection(vectors) {
  const center = new THREE.Vector3();
  vectors.forEach((vector) => center.add(vector.clone().normalize()));
  if (center.lengthSq() < 1e-8) return vectors[0]?.clone().normalize() || new THREE.Vector3(0, 1, 0);
  return center.normalize();
}

export function regionLayout(course, region) {
  if (region.__layout) return region.__layout;
  const boundary = regionBoundaryVectors(course, region, 1);
  const parent = region.chapterId
    ? course.chapters.find((chapter) => chapter.id === region.chapterId)
    : null;
  const parentBoundary = parent ? regionBoundaryVectors(course, parent, 1) : null;
  const denseBoundary = sampleClosedBoundary(course, region, 7, 1);
  const vertexCenter = centroidDirection(boundary);
  const isInsideRegion = (candidate) => sphericalContains(candidate, boundary)
    && (!parentBoundary || sphericalContains(candidate, parentBoundary));

  // Fibonacci 点在球面上近似等面积分布。对落在区域内部的点求平均，得到的
  // 是区域的面积中心，比“离边界最远点”更符合标签视觉居中的预期。
  const interiorSamples = fibonacciCandidates.filter(isInsideRegion);
  const areaCenter = centroidDirection(interiorSamples.length >= 3 ? interiorSamples : boundary);
  const candidates = [areaCenter, vertexCenter];
  boundary.forEach((point) => {
    candidates.push(slerpUnit(areaCenter, point, 0.22));
    candidates.push(slerpUnit(areaCenter, point, 0.44));
  });
  // 全球候选点让凹多边形也能找到真正位于内部的标签锚点。
  // 仅按顶点质心附近搜索时，质心落在凹口外会把标签放错区域。
  candidates.push(...interiorSamples);

  const evaluated = [];
  let bestClearance = -1;
  candidates.forEach((candidate) => {
    if (!isInsideRegion(candidate)) return;
    let clearance = Infinity;
    denseBoundary.forEach((edgePoint) => {
      clearance = Math.min(clearance, candidate.angleTo(edgePoint));
    });
    evaluated.push({ candidate, clearance });
    bestClearance = Math.max(bestClearance, clearance);
  });

  // 面积中心若靠近凹口或窄边，向区域内部做有限调整；在仍有一半最大边距的
  // 候选点中优先选择最接近面积中心的位置，而不是直接移到最大内切圆中心。
  const minimumClearance = Math.max(0.012, bestClearance * 0.5);
  const best = evaluated
    .filter((entry) => entry.clearance >= minimumClearance)
    .sort((a, b) => (
      a.candidate.angleTo(areaCenter) - b.candidate.angleTo(areaCenter)
      || b.clearance - a.clearance
    ))[0];

  region.__layout = {
    anchor: best?.candidate.clone().normalize() || vertexCenter,
    clearance: Math.max(0.035, best?.clearance || (bestClearance > 0 ? bestClearance : 0.08))
  };
  return region.__layout;
}

export function invalidateCourseLayouts(course) {
  [...course.chapters, ...course.sections].forEach((region) => {
    delete region.__layout;
  });
}

function createCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = FIELD_WIDTH;
  canvas.height = FIELD_HEIGHT;
  return canvas;
}

function createCanvasTexture(canvas, discrete = false) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  // 语义色图的每个像素都是离散的区域 ID。mipmap 会把相邻区域的颜色
  // 预先平均，在斜视角下表现为同一区域出现脏色或条纹。
  texture.minFilter = discrete ? THREE.NearestFilter : THREE.LinearFilter;
  texture.magFilter = discrete ? THREE.NearestFilter : THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.anisotropy = 1;
  return texture;
}

let directionGrid = null;

function getDirectionGrid() {
  if (directionGrid) return directionGrid;
  directionGrid = new Float32Array(FIELD_WIDTH * FIELD_HEIGHT * 3);
  for (let y = 0; y < FIELD_HEIGHT; y += 1) {
    const v = 1 - (y + 0.5) / FIELD_HEIGHT;
    const latitude = (v - 0.5) * Math.PI;
    const cosLatitude = Math.cos(latitude);
    const sinLatitude = Math.sin(latitude);
    for (let x = 0; x < FIELD_WIDTH; x += 1) {
      const phi = ((x + 0.5) / FIELD_WIDTH) * TAU;
      const offset = (y * FIELD_WIDTH + x) * 3;
      directionGrid[offset] = -Math.cos(phi) * cosLatitude;
      directionGrid[offset + 1] = sinLatitude;
      directionGrid[offset + 2] = Math.sin(phi) * cosLatitude;
    }
  }
  return directionGrid;
}

function colorBytes(color) {
  const hex = color.getHex(THREE.SRGBColorSpace);
  return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
}

// 用球面绕数而不是经纬平面多边形判断区域。这样不会受 UV 接缝和极点退化影响。
function sphericalContainsGridPoint(grid, offset, boundary, interiorSign) {
  const px = grid[offset];
  const py = grid[offset + 1];
  const pz = grid[offset + 2];
  let winding = 0;

  for (let index = 0; index < boundary.length; index += 1) {
    const a = boundary[index];
    const b = boundary[(index + 1) % boundary.length];
    const aDot = a.x * px + a.y * py + a.z * pz;
    const bDot = b.x * px + b.y * py + b.z * pz;
    const ax = a.x - px * aDot;
    const ay = a.y - py * aDot;
    const az = a.z - pz * aDot;
    const bx = b.x - px * bDot;
    const by = b.y - py * bDot;
    const bz = b.z - pz * bDot;
    const aLength = Math.hypot(ax, ay, az);
    const bLength = Math.hypot(bx, by, bz);
    if (aLength < 1e-6 || bLength < 1e-6) {
      return aDot > 0.999999 || bDot > 0.999999;
    }

    const invALength = 1 / aLength;
    const invBLength = 1 / bLength;
    const nax = ax * invALength;
    const nay = ay * invALength;
    const naz = az * invALength;
    const nbx = bx * invBLength;
    const nby = by * invBLength;
    const nbz = bz * invBLength;
    const crossX = nay * nbz - naz * nby;
    const crossY = naz * nbx - nax * nbz;
    const crossZ = nax * nby - nay * nbx;
    winding += Math.atan2(px * crossX + py * crossY + pz * crossZ, nax * nbx + nay * nby + naz * nbz);
  }

  return winding * interiorSign > Math.PI;
}

function buildRegionInfos(course, regions) {
  return regions.map((region) => {
    const boundary = regionBoundaryVectors(course, region, 1);
    if (boundary.length < 3) {
      throw new Error(`区域 ${region.id || region.title || 'unknown'} 至少需要三个有效顶点`);
    }
    const site = regionLayout(course, region).anchor;
    const interiorSign = boundaryInteriorSign(boundary);
    // 包围帽只用于跳过显然不可能命中的区域；实际归属仍由球面绕数确定。
    const sampledBoundary = sampleClosedBoundary(course, region, 8, 1);
    const capRadius = Math.min(
      Math.PI,
      Math.max(...sampledBoundary.map((point) => site.angleTo(point))) + 0.03
    );
    return { region, boundary, site, interiorSign, capCosine: Math.cos(capRadius) };
  });
}

function nearestRegionIndex(infos, px, py, pz, candidateIndices) {
  let bestIndex = candidateIndices[0];
  let bestDot = -Infinity;
  for (const index of candidateIndices) {
    const site = infos[index].site;
    const dot = px * site.x + py * site.y + pz * site.z;
    if (dot > bestDot) {
      bestDot = dot;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function chooseRegionIndex(infos, directions, offset, candidateIndices) {
  const px = directions[offset];
  const py = directions[offset + 1];
  const pz = directions[offset + 2];
  let matchedRegion = -1;
  let matchedDot = -Infinity;

  // 数据边界出现轻微重叠时，选择中心方向最近的区域，避免数组顺序决定归属。
  for (const index of candidateIndices) {
    const info = infos[index];
    const siteDot = px * info.site.x + py * info.site.y + pz * info.site.z;
    if (siteDot < info.capCosine) continue;
    if (sphericalContainsGridPoint(directions, offset, info.boundary, info.interiorSign)
      && siteDot > matchedDot) {
      matchedRegion = index;
      matchedDot = siteDot;
    }
  }

  // 包围帽只是加速条件；不命中时完整复核，避免凹区域或异常数据被误跳过。
  if (matchedRegion < 0) {
    for (const index of candidateIndices) {
      const info = infos[index];
      const siteDot = px * info.site.x + py * info.site.y + pz * info.site.z;
      if (sphericalContainsGridPoint(directions, offset, info.boundary, info.interiorSign)
        && siteDot > matchedDot) {
        matchedRegion = index;
        matchedDot = siteDot;
      }
    }
  }

  return matchedRegion >= 0
    ? matchedRegion
    : nearestRegionIndex(infos, px, py, pz, candidateIndices);
}

function smoothRegionIdEdges(sourceIds, passes = 2) {
  let current = sourceIds;
  for (let pass = 0; pass < passes; pass += 1) {
    const next = current.slice();
    for (let y = 0; y < FIELD_HEIGHT; y += 1) {
      for (let x = 0; x < FIELD_WIDTH; x += 1) {
        const pixel = y * FIELD_WIDTH + x;
        const ownId = current[pixel];
        const left = current[y * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH)];
        const right = current[y * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH)];
        const up = current[Math.max(0, y - 1) * FIELD_WIDTH + x];
        const down = current[Math.min(FIELD_HEIGHT - 1, y + 1) * FIELD_WIDTH + x];
        let replacement = -1;
        if (left === right && (left === up || left === down)) replacement = left;
        else if (left === up && left === down) replacement = left;
        else if (right === up && right === down) replacement = right;
        // 只修正被同一区域从至少三个方向包围的单像素凹口或尖刺。
        if (replacement >= 0 && replacement !== ownId) next[pixel] = replacement;
      }
    }
    current = next;
  }
  return current;
}

function mergeSmallRegionIdIslands(sourceIds, regionCount) {
  const componentIds = new Int32Array(sourceIds.length);
  componentIds.fill(-1);
  const queue = new Int32Array(sourceIds.length);
  const componentRegions = [];
  const componentSizes = [];
  const largestComponents = new Int32Array(regionCount);
  const largestSizes = new Int32Array(regionCount);
  largestComponents.fill(-1);

  for (let start = 0; start < sourceIds.length; start += 1) {
    if (componentIds[start] >= 0) continue;
    const componentId = componentSizes.length;
    const regionId = sourceIds[start];
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    componentIds[start] = componentId;

    while (head < tail) {
      const pixel = queue[head++];
      const y = Math.floor(pixel / FIELD_WIDTH);
      const x = pixel - y * FIELD_WIDTH;
      const left = y * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH);
      const right = y * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH);
      if (componentIds[left] < 0 && sourceIds[left] === regionId) {
        componentIds[left] = componentId;
        queue[tail++] = left;
      }
      if (componentIds[right] < 0 && sourceIds[right] === regionId) {
        componentIds[right] = componentId;
        queue[tail++] = right;
      }
      if (y > 0) {
        const up = pixel - FIELD_WIDTH;
        if (componentIds[up] < 0 && sourceIds[up] === regionId) {
          componentIds[up] = componentId;
          queue[tail++] = up;
        }
      }
      if (y + 1 < FIELD_HEIGHT) {
        const down = pixel + FIELD_WIDTH;
        if (componentIds[down] < 0 && sourceIds[down] === regionId) {
          componentIds[down] = componentId;
          queue[tail++] = down;
        }
      }
    }

    componentRegions.push(regionId);
    componentSizes.push(tail);
    if (tail > largestSizes[regionId]) {
      largestSizes[regionId] = tail;
      largestComponents[regionId] = componentId;
    }
  }

  // 章节区域按定义应为单一连通面。仅归并占整张图不到千分之一的次级
  // 连通块，既去掉重叠边界产生的浮岛，也保留主体和真实的窄区域。
  const maximumIslandSize = Math.max(12, Math.round(sourceIds.length * 0.001));
  const borderCounts = componentSizes.map((size, componentId) => (
    componentId !== largestComponents[componentRegions[componentId]] && size <= maximumIslandSize
      ? new Map()
      : null
  ));

  for (let pixel = 0; pixel < sourceIds.length; pixel += 1) {
    const componentId = componentIds[pixel];
    const counts = borderCounts[componentId];
    if (!counts) continue;
    const y = Math.floor(pixel / FIELD_WIDTH);
    const x = pixel - y * FIELD_WIDTH;
    const neighbors = [
      y * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH),
      y * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH)
    ];
    if (y > 0) neighbors.push(pixel - FIELD_WIDTH);
    if (y + 1 < FIELD_HEIGHT) neighbors.push(pixel + FIELD_WIDTH);
    neighbors.forEach((neighbor) => {
      const neighborId = sourceIds[neighbor];
      if (neighborId !== sourceIds[pixel]) {
        counts.set(neighborId, (counts.get(neighborId) || 0) + 1);
      }
    });
  }

  const replacements = borderCounts.map((counts) => {
    if (!counts?.size) return -1;
    let replacement = -1;
    let bestCount = -1;
    counts.forEach((count, regionId) => {
      if (count > bestCount) {
        replacement = regionId;
        bestCount = count;
      }
    });
    return replacement;
  });
  const cleaned = sourceIds.slice();
  for (let pixel = 0; pixel < cleaned.length; pixel += 1) {
    const replacement = replacements[componentIds[pixel]];
    if (replacement >= 0) cleaned[pixel] = replacement;
  }
  return cleaned;
}

function buildSphericalIds(course, regions, parentIds = null) {
  const infos = buildRegionInfos(course, regions);
  const directions = getDirectionGrid();
  const ids = new Int16Array(FIELD_WIDTH * FIELD_HEIGHT);
  const allIndices = infos.map((_, index) => index);
  const indicesByChapter = new Map();
  if (parentIds) {
    infos.forEach((info, index) => {
      const siblings = indicesByChapter.get(info.region.chapterId) || [];
      siblings.push(index);
      indicesByChapter.set(info.region.chapterId, siblings);
    });
  }

  for (let pixel = 0; pixel < ids.length; pixel += 1) {
    const offset = pixel * 3;
    const chapterId = parentIds ? course.chapters[parentIds[pixel]]?.id : null;
    const candidateIndices = (chapterId && indicesByChapter.get(chapterId)) || allIndices;
    ids[pixel] = chooseRegionIndex(infos, directions, offset, candidateIndices);
  }

  if (regions !== course.chapters) return ids;
  const cleaned = mergeSmallRegionIdIslands(smoothRegionIdEdges(ids), regions.length);
  // 点击检测复用同一份 ID，避免清理后的视觉边界与章节命中边界不一致。
  course.__chapterRegionIds = cleaned;
  return cleaned;
}

function buildField(course, regions, level, ids = buildSphericalIds(course, regions)) {
  const pixelCount = FIELD_WIDTH * FIELD_HEIGHT;

  const distance = new Float32Array(pixelCount);
  distance.fill(1e6);
  for (let y = 0; y < FIELD_HEIGHT; y += 1) {
    for (let x = 0; x < FIELD_WIDTH; x += 1) {
      const pixel = y * FIELD_WIDTH + x;
      const left = y * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH);
      const right = y * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH);
      const up = Math.max(0, y - 1) * FIELD_WIDTH + x;
      const down = Math.min(FIELD_HEIGHT - 1, y + 1) * FIELD_WIDTH + x;
      if (ids[pixel] !== ids[left] || ids[pixel] !== ids[right] || ids[pixel] !== ids[up] || ids[pixel] !== ids[down]) {
        distance[pixel] = 0;
      }
    }
  }

  for (let pass = 0; pass < 2; pass += 1) {
    for (let y = 0; y < FIELD_HEIGHT; y += 1) {
      for (let x = 0; x < FIELD_WIDTH; x += 1) {
        const pixel = y * FIELD_WIDTH + x;
        const left = y * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH);
        const up = Math.max(0, y - 1) * FIELD_WIDTH + x;
        const upLeft = Math.max(0, y - 1) * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH);
        const upRight = Math.max(0, y - 1) * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH);
        distance[pixel] = Math.min(
          distance[pixel],
          distance[left] + 1,
          distance[up] + 1,
          distance[upLeft] + 1.414,
          distance[upRight] + 1.414
        );
      }
    }
    for (let y = FIELD_HEIGHT - 1; y >= 0; y -= 1) {
      for (let x = FIELD_WIDTH - 1; x >= 0; x -= 1) {
        const pixel = y * FIELD_WIDTH + x;
        const right = y * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH);
        const down = Math.min(FIELD_HEIGHT - 1, y + 1) * FIELD_WIDTH + x;
        const downLeft = Math.min(FIELD_HEIGHT - 1, y + 1) * FIELD_WIDTH + ((x - 1 + FIELD_WIDTH) % FIELD_WIDTH);
        const downRight = Math.min(FIELD_HEIGHT - 1, y + 1) * FIELD_WIDTH + ((x + 1) % FIELD_WIDTH);
        distance[pixel] = Math.min(
          distance[pixel],
          distance[right] + 1,
          distance[down] + 1,
          distance[downLeft] + 1.414,
          distance[downRight] + 1.414
        );
      }
    }
  }

  const colorCanvas = createCanvas();
  const heightCanvas = createCanvas();
  const colorContext = colorCanvas.getContext('2d');
  const heightContext = heightCanvas.getContext('2d');
  const colorImage = colorContext.createImageData(FIELD_WIDTH, FIELD_HEIGHT);
  const heightImage = heightContext.createImageData(FIELD_WIDTH, FIELD_HEIGHT);
  const palette = regions.map((region) => colorBytes(region.__color || new THREE.Color(course.color)));
  const seamWidth = level === 'chapter' ? 9 : 4.5;
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const offset = pixel * 4;
    const [red, green, blue] = palette[ids[pixel]];
    colorImage.data[offset] = red;
    colorImage.data[offset + 1] = green;
    colorImage.data[offset + 2] = blue;
    colorImage.data[offset + 3] = 255;
    const normalizedDistance = THREE.MathUtils.clamp((distance[pixel] - 0.4) / seamWidth, 0, 1);
    const eased = normalizedDistance * normalizedDistance * (3 - 2 * normalizedDistance);
    const heightValue = Math.round(12 + eased * 243);
    heightImage.data[offset] = heightValue;
    heightImage.data[offset + 1] = heightValue;
    heightImage.data[offset + 2] = heightValue;
    heightImage.data[offset + 3] = 255;
  }
  colorContext.putImageData(colorImage, 0, 0);
  heightContext.putImageData(heightImage, 0, 0);

  return {
    color: createCanvasTexture(colorCanvas, true),
    height: createCanvasTexture(heightCanvas)
  };
}

function createSolidField(color) {
  const colorCanvas = createCanvas();
  const heightCanvas = createCanvas();
  const colorContext = colorCanvas.getContext('2d');
  const heightContext = heightCanvas.getContext('2d');
  colorContext.fillStyle = color;
  colorContext.fillRect(0, 0, FIELD_WIDTH, FIELD_HEIGHT);
  heightContext.fillStyle = '#ffffff';
  heightContext.fillRect(0, 0, FIELD_WIDTH, FIELD_HEIGHT);
  return {
    color: createCanvasTexture(colorCanvas, true),
    height: createCanvasTexture(heightCanvas)
  };
}

export function createSemanticFields(course) {
  if (!course.chapters.length) {
    return {
      chapter: createSolidField(course.color),
      section: createSolidField(course.color)
    };
  }
  const chapterIds = buildSphericalIds(course, course.chapters);
  const sectionRegions = course.sections;
  const sectionIds = sectionRegions.length
    ? buildSphericalIds(course, sectionRegions, chapterIds)
    : null;
  return {
    chapter: buildField(course, course.chapters, 'chapter', chapterIds),
    section: sectionRegions.length
      ? buildField(course, sectionRegions, 'section', sectionIds)
      : createSolidField(course.color)
  };
}

export function disposeSemanticFields(fields) {
  fields?.chapter?.color?.dispose();
  fields?.chapter?.height?.dispose();
  fields?.section?.color?.dispose();
  fields?.section?.height?.dispose();
}

export function createSemanticMaterial(fields, courseColor) {
  return new THREE.ShaderMaterial({
    uniforms: {
      uChapterMap: { value: fields.chapter.color },
      uSectionMap: { value: fields.section.color },
      uChapterHeight: { value: fields.chapter.height },
      uSectionHeight: { value: fields.section.height },
      uBaseColor: { value: lighten(courseColor, 0.44) },
      uRegionReveal: { value: 0 },
      uSectionBlend: { value: 0 },
      uDisplacement: { value: 0.046 },
      uBumpStrength: { value: 0.42 },
      uTexel: { value: new THREE.Vector2(1 / FIELD_WIDTH, 1 / FIELD_HEIGHT) }
    },
    vertexShader: `
      uniform sampler2D uChapterHeight;
      uniform sampler2D uSectionHeight;
      uniform float uRegionReveal;
      uniform float uSectionBlend;
      uniform float uDisplacement;
      varying vec2 vUvField;
      varying vec3 vWorldPosition;
      varying vec3 vWorldNormal;

      float combinedHeight(float chapterHeight, float sectionHeight) {
        float chapterGroove = 1.0 - chapterHeight;
        float sectionGroove = (1.0 - sectionHeight) * uSectionBlend * 0.62;
        return 1.0 - max(chapterGroove, sectionGroove);
      }

      void main() {
        float chapterHeight = texture2D(uChapterHeight, uv).r;
        float sectionHeight = texture2D(uSectionHeight, uv).r;
        float fieldHeight = combinedHeight(chapterHeight, sectionHeight);
        float displacement = (fieldHeight - 0.78) * uDisplacement * uRegionReveal;
        vec3 transformed = position + normal * displacement;
        vec4 worldPosition = modelMatrix * vec4(transformed, 1.0);
        vUvField = uv;
        vWorldPosition = worldPosition.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader: `
      uniform sampler2D uChapterMap;
      uniform sampler2D uSectionMap;
      uniform sampler2D uChapterHeight;
      uniform sampler2D uSectionHeight;
      uniform vec3 uBaseColor;
      uniform float uRegionReveal;
      uniform float uSectionBlend;
      uniform float uBumpStrength;
      uniform vec2 uTexel;
      varying vec2 vUvField;
      varying vec3 vWorldPosition;
      varying vec3 vWorldNormal;

      vec3 srgbToLinear(vec3 value) {
        vec3 low = value / 12.92;
        vec3 high = pow((value + 0.055) / 1.055, vec3(2.4));
        return mix(low, high, step(vec3(0.04045), value));
      }

      float fieldHeightAt(vec2 uv) {
        float chapterHeight = texture2D(uChapterHeight, uv).r;
        float sectionHeight = texture2D(uSectionHeight, uv).r;
        float chapterGroove = 1.0 - chapterHeight;
        float sectionGroove = (1.0 - sectionHeight) * uSectionBlend * 0.62;
        return 1.0 - max(chapterGroove, sectionGroove);
      }

      float surfaceNoise(vec3 point) {
        return fract(sin(dot(point, vec3(12.9898, 78.233, 41.164))) * 43758.5453);
      }

      void main() {
        vec3 chapterColor = srgbToLinear(texture2D(uChapterMap, vUvField).rgb);
        vec3 sectionColor = srgbToLinear(texture2D(uSectionMap, vUvField).rgb);
        float chapterHeight = texture2D(uChapterHeight, vUvField).r;
        float sectionHeight = texture2D(uSectionHeight, vUvField).r;
        vec3 surfaceColor = mix(uBaseColor, chapterColor, uRegionReveal);
        surfaceColor = mix(surfaceColor, sectionColor, uRegionReveal * uSectionBlend);

        float chapterSeam = (1.0 - smoothstep(0.22, 0.86, chapterHeight)) * uRegionReveal;
        float sectionSeam = (1.0 - smoothstep(0.12, 0.72, sectionHeight))
          * uRegionReveal * uSectionBlend * (1.0 - chapterSeam * 0.72);
        float chapterCore = (1.0 - smoothstep(0.05, 0.30, chapterHeight)) * uRegionReveal;
        float sectionCore = (1.0 - smoothstep(0.05, 0.27, sectionHeight))
          * uRegionReveal * uSectionBlend * (1.0 - chapterCore * 0.82);
        float seam = max(chapterSeam, sectionSeam);
        surfaceColor *= mix(1.0, 0.93, sectionSeam);
        surfaceColor *= mix(1.0, 0.82, chapterSeam);
        surfaceColor = mix(surfaceColor, vec3(0.035, 0.050, 0.065), sectionCore * 0.12);
        surfaceColor = mix(surfaceColor, vec3(0.025, 0.038, 0.052), chapterCore * 0.30);

        float hLeft = fieldHeightAt(vUvField - vec2(uTexel.x, 0.0));
        float hRight = fieldHeightAt(vUvField + vec2(uTexel.x, 0.0));
        float hDown = fieldHeightAt(vUvField - vec2(0.0, uTexel.y));
        float hUp = fieldHeightAt(vUvField + vec2(0.0, uTexel.y));
        vec3 normal = normalize(vWorldNormal);
        vec3 tangent = cross(vec3(0.0, 1.0, 0.0), normal);
        if (dot(tangent, tangent) < 0.01) tangent = cross(vec3(1.0, 0.0, 0.0), normal);
        tangent = normalize(tangent);
        vec3 bitangent = normalize(cross(normal, tangent));
        normal = normalize(normal + tangent * (hLeft - hRight) * uBumpStrength * uRegionReveal
          + bitangent * (hDown - hUp) * uBumpStrength * uRegionReveal);

        vec3 keyDirection = normalize(vec3(0.58, 0.78, 0.52));
        vec3 fillDirection = normalize(vec3(-0.48, 0.30, -0.62));
        vec3 viewDirection = normalize(cameraPosition - vWorldPosition);
        float key = max(dot(normal, keyDirection), 0.0);
        float fill = max(dot(normal, fillDirection), 0.0);
        float hemisphere = normal.y * 0.5 + 0.5;
        vec3 halfDirection = normalize(keyDirection + viewDirection);
        float specular = pow(max(dot(normal, halfDirection), 0.0), 34.0) * (1.0 - seam * 0.7);
        float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 2.4);
        float grain = (surfaceNoise(vWorldPosition * 18.0) - 0.5) * 0.004;

        float light = 0.74 + key * 0.24 + fill * 0.07 + hemisphere * 0.04;
        vec3 finalColor = surfaceColor * (light + grain);
        finalColor += vec3(1.0, 0.96, 0.90) * specular * 0.09;
        finalColor += vec3(0.30, 0.54, 0.70) * rim * 0.035;
        gl_FragColor = vec4(finalColor, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
    side: THREE.FrontSide,
    transparent: false,
    depthWrite: true,
    depthTest: true,
    toneMapped: true
  });
}

export function updateMaterialFields(material, fields) {
  material.uniforms.uChapterMap.value = fields.chapter.color;
  material.uniforms.uSectionMap.value = fields.section.color;
  material.uniforms.uChapterHeight.value = fields.chapter.height;
  material.uniforms.uSectionHeight.value = fields.section.height;
}
