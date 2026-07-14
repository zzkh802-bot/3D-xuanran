import * as THREE from 'three';

const TAU = Math.PI * 2;
const FIELD_WIDTH = 768;
const FIELD_HEIGHT = 384;

const CHAPTER_PALETTE = [
  '#ef3f4f', '#16a3d6', '#55b73f', '#ff9f1c',
  '#8c52d6', '#eb5e28', '#00a896', '#3f6ad8',
  '#db4fa4', '#e3b341', '#2d9c5a', '#6b5bd2'
];

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
  course.chapters.forEach((chapter, chapterIndex) => {
    chapter.__color = new THREE.Color(CHAPTER_PALETTE[chapterIndex % CHAPTER_PALETTE.length]);
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

export function sphericalContains(point, boundary) {
  if (boundary.length < 3) return false;
  const direction = point.clone().normalize();
  let winding = 0;
  for (let index = 0; index < boundary.length; index += 1) {
    const a = boundary[index].clone().normalize();
    const b = boundary[(index + 1) % boundary.length].clone().normalize();
    const tangentA = a.addScaledVector(direction, -a.dot(direction));
    const tangentB = b.addScaledVector(direction, -b.dot(direction));
    if (tangentA.lengthSq() < 1e-10 || tangentB.lengthSq() < 1e-10) return true;
    tangentA.normalize();
    tangentB.normalize();
    const cross = new THREE.Vector3().crossVectors(tangentA, tangentB);
    winding += Math.atan2(direction.dot(cross), tangentA.dot(tangentB));
  }
  return Math.abs(winding) > Math.PI;
}

export function findRegionAtDirection(course, regions, direction) {
  for (let index = regions.length - 1; index >= 0; index -= 1) {
    const boundary = regionBoundaryVectors(course, regions[index], 1);
    if (sphericalContains(direction, boundary)) return regions[index];
  }
  return null;
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
  const denseBoundary = sampleClosedBoundary(course, region, 7, 1);
  const center = centroidDirection(boundary);
  const candidates = [center];
  boundary.forEach((point) => {
    candidates.push(slerpUnit(center, point, 0.22));
    candidates.push(slerpUnit(center, point, 0.44));
  });
  const cap = Math.min(Math.PI, Math.max(...boundary.map((point) => center.angleTo(point)), 0.15) + 0.2);
  const capCos = Math.cos(cap);
  fibonacciCandidates.forEach((candidate) => {
    if (candidate.dot(center) >= capCos) candidates.push(candidate);
  });

  let best = null;
  let bestClearance = -1;
  candidates.forEach((candidate) => {
    if (!sphericalContains(candidate, boundary)) return;
    let clearance = Infinity;
    denseBoundary.forEach((edgePoint) => {
      clearance = Math.min(clearance, candidate.angleTo(edgePoint));
    });
    if (clearance > bestClearance) {
      best = candidate.clone().normalize();
      bestClearance = clearance;
    }
  });

  region.__layout = {
    anchor: best || center,
    clearance: Math.max(0.035, bestClearance > 0 ? bestClearance : 0.08)
  };
  return region.__layout;
}

export function invalidateCourseLayouts(course) {
  [...course.chapters, ...course.sections].forEach((region) => {
    delete region.__layout;
  });
}

function vectorToUv(vector) {
  const direction = vector.clone().normalize();
  let u = Math.atan2(direction.z, -direction.x) / TAU;
  if (u < 0) u += 1;
  const v = Math.asin(THREE.MathUtils.clamp(direction.y, -1, 1)) / Math.PI + 0.5;
  return { u, v };
}

function unwrapUvs(vectors) {
  const points = vectors.map(vectorToUv);
  for (let index = 1; index < points.length; index += 1) {
    while (points[index].u - points[index - 1].u > 0.5) points[index].u -= 1;
    while (points[index].u - points[index - 1].u < -0.5) points[index].u += 1;
  }
  return points;
}

function tracePath(context, points, shift) {
  context.beginPath();
  points.forEach((point, index) => {
    const x = (point.u + shift) * FIELD_WIDTH;
    const y = (1 - point.v) * FIELD_HEIGHT;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.closePath();
}

function drawRepeated(context, points, operation) {
  for (let shift = -2; shift <= 2; shift += 1) {
    tracePath(context, points, shift);
    operation(context);
  }
}

function createCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = FIELD_WIDTH;
  canvas.height = FIELD_HEIGHT;
  return canvas;
}

function createCanvasTexture(canvas, renderer) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
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

function regionSite(course, region) {
  const vectors = regionBoundaryVectors(course, region, 1);
  return centroidDirection(vectors);
}

function colorBytes(color) {
  const hex = color.getHex(THREE.SRGBColorSpace);
  return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
}

function regionCode(index) {
  return [
    24 + ((index * 83 + 47) % 208),
    24 + ((index * 151 + 91) % 208),
    24 + ((index * 197 + 139) % 208)
  ];
}

function cleanupIdIslands(sourceIds) {
  let current = sourceIds;
  const counts = new Uint8Array(256);
  const touched = new Int16Array(8);
  const neighbors = new Int16Array(8);
  for (let pass = 0; pass < 2; pass += 1) {
    const next = current.slice();
    for (let y = 0; y < FIELD_HEIGHT; y += 1) {
      const upY = Math.max(0, y - 1);
      const downY = Math.min(FIELD_HEIGHT - 1, y + 1);
      for (let x = 0; x < FIELD_WIDTH; x += 1) {
        const leftX = (x - 1 + FIELD_WIDTH) % FIELD_WIDTH;
        const rightX = (x + 1) % FIELD_WIDTH;
        const pixel = y * FIELD_WIDTH + x;
        neighbors[0] = current[upY * FIELD_WIDTH + leftX];
        neighbors[1] = current[upY * FIELD_WIDTH + x];
        neighbors[2] = current[upY * FIELD_WIDTH + rightX];
        neighbors[3] = current[y * FIELD_WIDTH + leftX];
        neighbors[4] = current[y * FIELD_WIDTH + rightX];
        neighbors[5] = current[downY * FIELD_WIDTH + leftX];
        neighbors[6] = current[downY * FIELD_WIDTH + x];
        neighbors[7] = current[downY * FIELD_WIDTH + rightX];
        let touchedCount = 0;
        for (let index = 0; index < 8; index += 1) {
          const value = neighbors[index];
          if (counts[value] === 0) touched[touchedCount++] = value;
          counts[value] += 1;
        }
        const ownId = current[pixel];
        const ownCount = counts[ownId];
        let bestId = ownId;
        let bestCount = ownCount;
        if (ownCount < 3) {
          for (let index = 0; index < touchedCount; index += 1) {
            const candidate = touched[index];
            if (counts[candidate] > bestCount) {
              bestId = candidate;
              bestCount = counts[candidate];
            }
          }
          if (bestId !== ownId && bestCount >= 4) next[pixel] = bestId;
        }
        for (let index = 0; index < touchedCount; index += 1) counts[touched[index]] = 0;
      }
    }
    current = next;
  }
  return current;
}

function buildField(course, regions, renderer, level) {
  const idCanvas = createCanvas();
  const idContext = idCanvas.getContext('2d', { willReadFrequently: true });
  const paths = regions.map((region) => ({
    region,
    points: unwrapUvs(sampleClosedBoundary(course, region, level === 'chapter' ? 12 : 9, 1))
  }));
  idContext.clearRect(0, 0, FIELD_WIDTH, FIELD_HEIGHT);
  const codes = regions.map((_, index) => regionCode(index));
  paths.forEach(({ points }, index) => {
    const [red, green, blue] = codes[index];
    idContext.fillStyle = `rgb(${red}, ${green}, ${blue})`;
    drawRepeated(idContext, points, (context) => context.fill('nonzero'));
  });

  const pixelCount = FIELD_WIDTH * FIELD_HEIGHT;
  const sourceData = idContext.getImageData(0, 0, FIELD_WIDTH, FIELD_HEIGHT).data;
  let ids = new Int16Array(pixelCount);
  ids.fill(-1);
  const exactCodes = new Map(codes.map((code, index) => [
    (code[0] << 16) | (code[1] << 8) | code[2],
    index
  ]));
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const offset = pixel * 4;
    if (sourceData[offset + 3] < 96) continue;
    const red = sourceData[offset];
    const green = sourceData[offset + 1];
    const blue = sourceData[offset + 2];
    const exact = exactCodes.get((red << 16) | (green << 8) | blue);
    if (exact !== undefined) {
      ids[pixel] = exact;
      continue;
    }
    let bestRegion = 0;
    let bestDistance = Infinity;
    for (let regionIndex = 0; regionIndex < codes.length; regionIndex += 1) {
      const code = codes[regionIndex];
      const distanceToCode = (red - code[0]) ** 2 + (green - code[1]) ** 2 + (blue - code[2]) ** 2;
      if (distanceToCode < bestDistance) {
        bestDistance = distanceToCode;
        bestRegion = regionIndex;
      }
    }
    ids[pixel] = bestRegion;
  }

  ids = cleanupIdIslands(ids);

  const sites = regions.map((region) => regionSite(course, region));
  const directions = getDirectionGrid();
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    if (ids[pixel] >= 0) continue;
    const offset = pixel * 3;
    let bestRegion = 0;
    let bestDot = -Infinity;
    for (let regionIndex = 0; regionIndex < sites.length; regionIndex += 1) {
      const site = sites[regionIndex];
      const dot = directions[offset] * site.x
        + directions[offset + 1] * site.y
        + directions[offset + 2] * site.z;
      if (dot > bestDot) {
        bestDot = dot;
        bestRegion = regionIndex;
      }
    }
    ids[pixel] = bestRegion;
  }

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
  const seamWidth = level === 'chapter' ? 8.5 : 5.5;
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
    color: createCanvasTexture(colorCanvas, renderer),
    height: createCanvasTexture(heightCanvas, renderer)
  };
}

export function createSemanticFields(course, renderer) {
  return {
    chapter: buildField(course, course.chapters, renderer, 'chapter'),
    section: buildField(course, course.sections, renderer, 'section')
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
      uDisplacement: { value: 0.11 },
      uBumpStrength: { value: 1.65 },
      uTexel: { value: new THREE.Vector2(2 / FIELD_WIDTH, 2 / FIELD_HEIGHT) }
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
      varying float vFieldHeight;

      void main() {
        float chapterHeight = texture2D(uChapterHeight, uv).r;
        float sectionHeight = texture2D(uSectionHeight, uv).r;
        float fieldHeight = mix(chapterHeight, sectionHeight, uSectionBlend);
        float displacement = (fieldHeight - 0.58) * uDisplacement * uRegionReveal;
        vec3 transformed = position + normal * displacement;
        vec4 worldPosition = modelMatrix * vec4(transformed, 1.0);
        vUvField = uv;
        vWorldPosition = worldPosition.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vFieldHeight = fieldHeight;
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
      varying float vFieldHeight;

      vec3 srgbToLinear(vec3 value) {
        vec3 low = value / 12.92;
        vec3 high = pow((value + 0.055) / 1.055, vec3(2.4));
        return mix(low, high, step(vec3(0.04045), value));
      }

      float fieldHeightAt(vec2 uv) {
        float chapterHeight = texture2D(uChapterHeight, uv).r;
        float sectionHeight = texture2D(uSectionHeight, uv).r;
        return mix(chapterHeight, sectionHeight, uSectionBlend);
      }

      float surfaceNoise(vec3 point) {
        return fract(sin(dot(point, vec3(12.9898, 78.233, 41.164))) * 43758.5453);
      }

      void main() {
        vec3 chapterColor = srgbToLinear(texture2D(uChapterMap, vUvField).rgb);
        vec3 sectionColor = srgbToLinear(texture2D(uSectionMap, vUvField).rgb);
        vec3 surfaceColor = mix(uBaseColor, chapterColor, uRegionReveal);
        surfaceColor = mix(surfaceColor, sectionColor, uRegionReveal * uSectionBlend);

        float seam = (1.0 - smoothstep(0.16, 0.80, vFieldHeight)) * uRegionReveal;
        float seamCore = (1.0 - smoothstep(0.05, 0.34, vFieldHeight)) * uRegionReveal;
        surfaceColor *= mix(1.0, 0.82, seam);
        surfaceColor = mix(surfaceColor, vec3(0.035, 0.050, 0.065), seamCore * 0.34);

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
        float grain = (surfaceNoise(vWorldPosition * 18.0) - 0.5) * 0.025;

        float light = 0.52 + key * 0.58 + fill * 0.16 + hemisphere * 0.10;
        vec3 finalColor = surfaceColor * (light + grain);
        finalColor += vec3(1.0, 0.96, 0.90) * specular * 0.24;
        finalColor += vec3(0.30, 0.54, 0.70) * rim * 0.08;
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
