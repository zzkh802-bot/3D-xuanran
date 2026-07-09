import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import data from './data/course-data.json';
import './styles.css';

const sceneEl = document.querySelector('#scene');
const tabsEl = document.querySelector('#courseTabs');
const lodLabel = document.querySelector('#lodLabel');
const courseTitle = document.querySelector('#courseTitle');
const selectionText = document.querySelector('#selectionText');
const overviewBtn = document.querySelector('#overviewBtn');
const editBtn = document.querySelector('#editBtn');
const editorEl = document.querySelector('#editor');
const phiInput = document.querySelector('#phiInput');
const psiInput = document.querySelector('#psiInput');

const radius = data.radius || 10;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef2f7);

const overviewDistance = 122;
const overviewSpacing = 26.5;
const focusDistance = 55;

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1200);
camera.position.set(0, 10, overviewDistance);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
sceneEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 14;
controls.maxDistance = 170;

scene.add(new THREE.HemisphereLight(0xffffff, 0x8ea0b8, 1.75));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.55);
keyLight.position.set(30, 35, 28);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xb8f1ff, 1.15);
rimLight.position.set(-34, 18, -26);
scene.add(rimLight);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const root = new THREE.Group();
scene.add(root);

const courseGroups = [];
let selectedCourse = null;
let selectedNode = null;
let editMode = false;
let draggingPoint = null;
let currentLodLevel = 0;

const chapterPalette = [
  '#e11d48', '#0891b2', '#65a30d', '#f59e0b',
  '#7c3aed', '#dc2626', '#0d9488', '#2563eb',
  '#c026d3', '#ea580c', '#16a34a', '#4f46e5'
];
const regionTextureCache = new Map();

function colorToThree(color, offset = 0) {
  const c = new THREE.Color(color);
  if (offset > 0) c.lerp(new THREE.Color(0xffffff), offset);
  if (offset < 0) c.lerp(new THREE.Color(0x101820), Math.abs(offset));
  return c;
}

function readableTextColor(color) {
  const c = new THREE.Color(color);
  const luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  return luminance < 0.48 ? '#ffffff' : '#102033';
}

function smoothstep(edge0, edge1, value) {
  const t = THREE.MathUtils.clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function chapterColor(index) {
  return chapterPalette[index % chapterPalette.length];
}

function sectionColor(baseColor, index, total) {
  const source = new THREE.Color(baseColor);
  const hsl = {};
  source.getHSL(hsl);
  const phase = total <= 1 ? 0.5 : index / (total - 1);
  const lightBands = [0.42, 0.66, 0.52, 0.76, 0.34, 0.58, 0.47, 0.70];
  const saturationBands = [0.92, 0.78, 0.98, 0.84, 0.88, 0.96, 0.74, 0.90];
  const color = new THREE.Color();
  const hue = (hsl.h + (phase - 0.5) * 0.055 + ((index % 3) - 1) * 0.012 + 1) % 1;
  const saturation = THREE.MathUtils.clamp(Math.max(hsl.s, 0.72) * saturationBands[index % saturationBands.length], 0.58, 0.98);
  const lightness = THREE.MathUtils.clamp(lightBands[index % lightBands.length] + (phase - 0.5) * 0.035, 0.30, 0.78);
  color.setHSL(hue, saturation, lightness);
  return color;
}

function hashNoise(x, y, seed) {
  const value = Math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453123;
  return value - Math.floor(value);
}

function makeRegionTexture(color, kind = 'region') {
  const base = new THREE.Color(color);
  const key = `${kind}:${base.getHexString()}`;
  if (regionTextureCache.has(key)) return regionTextureCache.get(key);

  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(size, size);
  const seed = parseInt(base.getHexString().slice(0, 5), 16) / 8192;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const nx = x / size - 0.5;
      const ny = y / size - 0.5;
      const grain =
        hashNoise(Math.floor(x / 5), Math.floor(y / 5), seed) * 0.032 +
        hashNoise(Math.floor(x / 17), Math.floor(y / 17), seed + 8.7) * 0.026;
      const contour = Math.sin((nx * 1.45 + ny * 1.05) * 21 + Math.sin((nx - ny) * 8)) * 0.018;
      const vignette = -Math.sqrt(nx * nx + ny * ny) * 0.045;
      const shade = THREE.MathUtils.clamp(0.985 + grain + contour + vignette, 0.88, 1.10);
      const idx = (y * size + x) * 4;
      image.data[idx] = THREE.MathUtils.clamp(base.r * 255 * shade, 0, 255);
      image.data[idx + 1] = THREE.MathUtils.clamp(base.g * 255 * shade, 0, 255);
      image.data[idx + 2] = THREE.MathUtils.clamp(base.b * 255 * shade, 0, 255);
      image.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.userData.sharedRegionTexture = true;
  regionTextureCache.set(key, texture);
  return texture;
}

function rememberOpacity(object) {
  object.traverse((child) => {
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    materials.forEach((material) => {
      if (material.userData.baseOpacity === undefined) {
        material.userData.baseOpacity = material.opacity ?? 1;
      }
      material.transparent = true;
    });
  });
  return object;
}

function setLayerOpacity(group, opacity) {
  group.visible = opacity > 0.015;
  group.traverse((child) => {
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    materials.forEach((material) => {
      if (material.userData.baseOpacity === undefined) {
        material.userData.baseOpacity = material.opacity ?? 1;
      }
      const baseOpacity = material.userData.baseOpacity;
      material.opacity = baseOpacity * opacity;
      material.transparent = true;
    });
  });
}

function sphericalToVector(phi, psi, r = radius) {
  const cp = Math.cos(psi);
  return new THREE.Vector3(
    r * cp * Math.cos(phi),
    r * Math.sin(psi),
    r * cp * Math.sin(phi)
  );
}

function sourceToVector(point, r = radius) {
  if (Number.isFinite(point?.x) && Number.isFinite(point?.y) && Number.isFinite(point?.z)) {
    const v = new THREE.Vector3(point.x, point.z, point.y);
    if (v.lengthSq() > 0.001) return v.normalize().multiplyScalar(r);
  }
  return sphericalToVector(point?.phi || 0, point?.psi || 0, r);
}

function vectorToSource(localVector, target) {
  const v = localVector.clone().normalize().multiplyScalar(radius);
  target.x = Number(v.x.toFixed(5));
  target.y = Number(v.z.toFixed(5));
  target.z = Number(v.y.toFixed(5));
  target.phi = Number(Math.atan2(v.z, v.x).toFixed(6));
  target.psi = Number(Math.asin(THREE.MathUtils.clamp(v.y / radius, -1, 1)).toFixed(6));
}

function isMobileView() {
  return window.innerWidth < 720;
}

function currentOverviewDistance() {
  return isMobileView() ? 150 : overviewDistance;
}

function overviewPosition(index) {
  if (!isMobileView()) return new THREE.Vector3((index - 1.5) * overviewSpacing, 0, 0);
  const col = index % 2 === 0 ? -1 : 1;
  const rowY = index < 2 ? 18 : -5;
  return new THREE.Vector3(col * 10.5, rowY, 0);
}

function wrapText(ctx, text, maxChars = 11) {
  const clean = String(text).replace(/\s+/g, ' ').trim();
  return clean.match(new RegExp(`.{1,${maxChars}}`, 'g')) || [clean];
}

function makeTextTexture(text, color = '#17202a') {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 640;
  canvas.height = 192;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = '600 38px "Microsoft YaHei", "Noto Sans SC", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.38)';
  ctx.shadowBlur = 8;
  ctx.shadowOffsetX = 3;
  ctx.shadowOffsetY = 4;
  ctx.lineJoin = 'round';
  ctx.lineWidth = color === '#ffffff' ? 7 : 5;
  ctx.strokeStyle = color === '#ffffff' ? 'rgba(8, 17, 32, 0.62)' : 'rgba(255, 255, 255, 0.68)';
  ctx.fillStyle = color;
  wrapText(ctx, text).slice(0, 3).forEach((line, idx, lines) => {
    const y = canvas.height / 2 + (idx - (lines.length - 1) / 2) * 44;
    ctx.strokeText(line, canvas.width / 2, y);
    ctx.fillText(line, canvas.width / 2, y);
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
  return texture;
}

function makeBillboardLabel(text, color = '#17202a', scale = 1) {
  const texture = makeTextTexture(text, color);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(6.1 * scale, 1.85 * scale, 1);
  return sprite;
}

function makeSurfaceLabel(text, color = '#17202a', scale = 1) {
  const texture = makeTextTexture(text, color);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -4
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4.8 * scale, 1.45 * scale), material);
  return mesh;
}

function fitLabelScale(text, vectors, minScale, maxScale) {
  if (!vectors.length) return minScale;
  const center = centroidOf(vectors, radius);
  const spread = Math.max(...vectors.map((v) => v.distanceTo(center)));
  const textPenalty = THREE.MathUtils.clamp(9 / Math.max(4, String(text).length), 0.54, 1);
  return THREE.MathUtils.clamp((spread / 4.6) * textPenalty, minScale, maxScale);
}

function labelAnchor(vectors, r = radius) {
  if (!vectors.length) return new THREE.Vector3(0, r, 0);
  const anchor = new THREE.Vector3();
  vectors.forEach((vector) => {
    const normal = vector.clone().normalize();
    const frontWeight = 0.35 + Math.max(0, normal.z) * 1.8;
    anchor.add(normal.multiplyScalar(frontWeight));
  });
  if (anchor.lengthSq() < 0.001) return centroidOf(vectors, r);
  return anchor.normalize().multiplyScalar(r);
}

function placeSurfaceObject(object, position, outward = 1.04) {
  const normal = position.clone().normalize();
  object.position.copy(normal.clone().multiplyScalar(radius * outward));
  object.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
}

function slerpUnit(a, b, t) {
  const an = a.clone().normalize();
  const bn = b.clone().normalize();
  const dot = THREE.MathUtils.clamp(an.dot(bn), -0.9995, 0.9995);
  const omega = Math.acos(dot);
  if (omega < 0.0001) return an.lerp(bn, t).normalize();
  const so = Math.sin(omega);
  return an.multiplyScalar(Math.sin((1 - t) * omega) / so).add(bn.multiplyScalar(Math.sin(t * omega) / so)).normalize();
}

function pathVectors(path, r = radius) {
  return (path || [])
    .map((point) => sourceToVector(point, r))
    .filter((point) => point.lengthSq() > 0.001);
}

function centroidOf(vectors, r = radius) {
  const center = new THREE.Vector3();
  vectors.forEach((v) => center.add(v.clone().normalize()));
  if (center.lengthSq() < 0.001) return new THREE.Vector3(0, r, 0);
  return center.normalize().multiplyScalar(r);
}

function createSphericalPatch(boundary, color, opacity = 1, rings = 8, lift = 1.012) {
  if (boundary.length < 3) return null;
  const normals = boundary.map((p) => p.clone().normalize());
  const center = centroidOf(boundary, 1).normalize();
  const vertices = [];
  const uvs = [];
  const push = (v) => vertices.push(v.x, v.y, v.z);
  const pushUv = (ring, index) => {
    if (ring === 0) {
      uvs.push(0.5, 0.5);
      return;
    }
    const angle = (index / normals.length) * Math.PI * 2;
    const dist = THREE.MathUtils.clamp(ring / rings, 0, 1) * 0.48;
    uvs.push(0.5 + Math.cos(angle) * dist, 0.5 + Math.sin(angle) * dist);
  };
  const pushVertex = (v, ring, index) => {
    push(v);
    pushUv(ring, index);
  };
  const reliefRadius = (t) => radius * (lift + 0.010 - Math.pow(t, 1.35) * 0.014);
  const ringPoint = (ring, index) => {
    const t = ring / rings;
    return slerpUnit(center, normals[index % normals.length], t).multiplyScalar(reliefRadius(t));
  };

  for (let i = 0; i < normals.length; i += 1) {
    pushVertex(center.clone().multiplyScalar(reliefRadius(0)), 0, i);
    pushVertex(ringPoint(1, i), 1, i);
    pushVertex(ringPoint(1, i + 1), 1, i + 1);
  }

  for (let ring = 1; ring < rings; ring += 1) {
    for (let i = 0; i < normals.length; i += 1) {
      const a = ringPoint(ring, i);
      const b = ringPoint(ring, i + 1);
      const c = ringPoint(ring + 1, i);
      const d = ringPoint(ring + 1, i + 1);
      pushVertex(a, ring, i); pushVertex(c, ring + 1, i); pushVertex(b, ring, i + 1);
      pushVertex(b, ring, i + 1); pushVertex(c, ring + 1, i); pushVertex(d, ring + 1, i + 1);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.computeVertexNormals();
  const texture = makeRegionTexture(color);
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: texture,
      bumpMap: texture,
      bumpScale: 0.026,
      roughness: 0.66,
      metalness: 0.015,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2
    })
  );
}

function createBoundaryLine(points, color, outward = 1.032, opacity = 0.28) {
  if (points.length < 2) return null;
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((p) => p.clone().normalize().multiplyScalar(radius * outward)));
  const line = new THREE.LineLoop(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity })
  );
  return line;
}

function createTubeOnSphere(points, color, tubeRadius, outward, opacity) {
  if (points.length < 4) return null;
  const curvePoints = points.map((p) => p.clone().normalize().multiplyScalar(radius * outward));
  const curve = new THREE.CatmullRomCurve3(curvePoints, true, 'centripetal', 0.22);
  const geometry = new THREE.TubeGeometry(curve, Math.max(72, points.length * 2), tubeRadius, 8, true);
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.86,
      metalness: 0.015,
      transparent: true,
      opacity,
      depthWrite: false
    })
  );
}

function createGroove(points, color = 0x273142, tubeRadius = 0.052) {
  if (points.length < 4) return null;
  const group = new THREE.Group();
  const softened = new THREE.Color(color).lerp(new THREE.Color(0x475569), 0.48).getHex();
  const shadow = createTubeOnSphere(points, 0x1f2937, tubeRadius * 1.55, 1.012, 0.22);
  const core = createTubeOnSphere(points, softened, tubeRadius, 1.026, 0.70);
  const glint = createBoundaryLine(points, 0xffffff, 1.033, 0.16);
  [shadow, core, glint].forEach((object) => {
    if (object) group.add(object);
  });
  return group.children.length ? group : null;
}

function sectionNeighborText(course, section) {
  const idx = course.sections.findIndex((item) => item.id === section?.id);
  const prev = idx > 0 ? course.sections[idx - 1].title : '本章起点';
  const next = idx >= 0 && idx < course.sections.length - 1 ? course.sections[idx + 1].title : '本章终点';
  return `前置：${prev}\n后接：${next}`;
}

function describeSelection(data) {
  if (!data?.type) return '未选择节点';
  if (data.type === 'course') return `教材：${data.course.title}`;
  if (data.type === 'chapter') return `章节：${data.chapter.title}\n小节数：${data.chapter.sectionIds?.length || 0}`;
  if (data.type === 'section') return `小节：${data.section.title}\n${sectionNeighborText(data.course, data.section)}`;
  if (data.type === 'knowledge') return `知识点：${data.section.title} / ${data.point.label}\n${sectionNeighborText(data.course, data.section)}`;
  if (data.type === 'control') return `可调交点：${data.section.title}\n拖动后边界线会同步变化`;
  if (data.type === 'point') return `外层点：${data.point.label}`;
  return '未选择节点';
}

function chapterIndexOf(course, chapterKey) {
  return Math.max(0, course.chapters.findIndex((chapter) => String(chapter.key) === String(chapterKey)));
}

function sectionOrdinal(course, section) {
  const siblings = course.sections.filter((item) => String(item.chapterKey) === String(section.chapterKey));
  return Math.max(0, siblings.findIndex((item) => item.id === section.id));
}

function sectionColorFor(course, section) {
  const chapterIndex = chapterIndexOf(course, section.chapterKey);
  const siblings = course.sections.filter((item) => String(item.chapterKey) === String(section.chapterKey));
  return sectionColor(chapterColor(chapterIndex), sectionOrdinal(course, section), siblings.length || 1);
}

function disposeObject(object) {
  if (!object) return;
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    materials.forEach((material) => {
      ['map', 'bumpMap', 'alphaMap', 'normalMap', 'roughnessMap'].forEach((key) => {
        if (material[key] && !material[key].userData?.sharedRegionTexture) material[key].dispose();
      });
      material.dispose();
    });
  });
}

function removeFromParent(object) {
  if (object?.parent) object.parent.remove(object);
  disposeObject(object);
}

function createSectionVisual(courseGroup, section, sectionIndex) {
  const vectors = pathVectors(section.path);
  const color = sectionColorFor(courseGroup.course, section);
  const objects = [];
  const patch = createSphericalPatch(vectors, color.getHex(), 1, 8, 1.018);
  if (patch) {
    patch.userData = { type: 'section', course: courseGroup.course, section };
    rememberOpacity(patch);
    courseGroup.sections.add(patch);
    objects.push(patch);
  }
  const groove = createGroove(vectors, 0x101820, 0.05);
  if (groove) {
    groove.userData = { type: 'section', course: courseGroup.course, section };
    rememberOpacity(groove);
    courseGroup.sections.add(groove);
    objects.push(groove);
  }
  const highlight = createBoundaryLine(vectors, 0xffffff);
  if (highlight) {
    highlight.material.opacity = 0.20;
    highlight.userData = { type: 'section', course: courseGroup.course, section };
    rememberOpacity(highlight);
    courseGroup.sections.add(highlight);
    objects.push(highlight);
  }
  if (vectors.length) {
    const scale = fitLabelScale(section.title, vectors, 0.52, 0.92);
    const label = makeSurfaceLabel(section.title, readableTextColor(color), scale);
    placeSurfaceObject(label, labelAnchor(vectors), 1.074);
    label.userData = { type: 'section', course: courseGroup.course, section };
    rememberOpacity(label);
    courseGroup.sectionLabels.add(label);
    objects.push(label);
  }
  section.__visualObjects = objects;
  section.__visualIndex = sectionIndex;
}

function rebuildSectionVisual(course, section) {
  const courseGroup = courseGroups.find((item) => item.course === course);
  if (!courseGroup || !section) return;
  (section.__visualObjects || []).forEach(removeFromParent);
  createSectionVisual(courseGroup, section, section.__visualIndex || course.sections.indexOf(section));
}

function buildCourse(course, index) {
  const group = new THREE.Group();
  group.position.copy(overviewPosition(index));
  root.add(group);

  const base = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 128, 96),
    new THREE.MeshStandardMaterial({
      color: colorToThree(course.color, 0.50),
      map: makeRegionTexture(colorToThree(course.color, 0.44).getHex(), 'base'),
      bumpMap: makeRegionTexture(colorToThree(course.color, 0.44).getHex(), 'base'),
      bumpScale: 0.028,
      roughness: 0.62,
      metalness: 0.015
    })
  );
  base.userData = { type: 'course', course };
  group.add(base);

  const outer = new THREE.Group();
  const chapters = new THREE.Group();
  const chapterLabels = new THREE.Group();
  const sections = new THREE.Group();
  const sectionLabels = new THREE.Group();
  const knowledge = new THREE.Group();
  const knowledgeLabels = new THREE.Group();
  const handles = new THREE.Group();
  group.add(outer, chapters, chapterLabels, sections, sectionLabels, knowledge, knowledgeLabels, handles);

  course.points.forEach((point, pointIndex) => {
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(pointIndex % 7 === 0 ? 0.23 : 0.15, 18, 14),
      new THREE.MeshStandardMaterial({ color: colorToThree(course.color, pointIndex % 7 === 0 ? -0.16 : 0.12) })
    );
    dot.position.copy(sourceToVector(point, radius * 1.095));
    dot.userData = { type: 'point', course, point };
    outer.add(dot);
  });

  course.chapters.forEach((chapter, chapterIndex) => {
    const vectors = pathVectors(chapter.path);
    const patchColor = chapterColor(chapterIndex);
    const patch = createSphericalPatch(vectors, new THREE.Color(patchColor).getHex(), 1, 10, 1.006);
    if (patch) {
      patch.userData = { type: 'chapter', course, chapter };
      rememberOpacity(patch);
      chapters.add(patch);
    }
    const groove = createGroove(vectors, 0x0f172a, 0.072);
    if (groove) {
      groove.userData = { type: 'chapter', course, chapter };
      rememberOpacity(groove);
      chapters.add(groove);
    }
    const line = createBoundaryLine(vectors, 0xffffff);
    if (line) {
      line.material.opacity = 0.22;
      rememberOpacity(line);
      chapters.add(line);
    }
    if (vectors.length) {
      const scale = fitLabelScale(chapter.title, vectors, 1.08, 1.72);
      const label = makeSurfaceLabel(chapter.title, readableTextColor(patchColor), scale);
      placeSurfaceObject(label, labelAnchor(vectors), 1.094);
      label.userData = { type: 'chapter', course, chapter };
      rememberOpacity(label);
      chapterLabels.add(label);
    }
  });

  course.sections.forEach((section, sectionIndex) => {
    createSectionVisual({ course, sections, sectionLabels }, section, sectionIndex);
    const editableStride = Math.max(12, Math.floor((section.path?.length || 1) / 7));
    (section.path || []).forEach((point, pointIndex) => {
      if (pointIndex % editableStride !== 0) return;
      const handle = new THREE.Mesh(
        new THREE.SphereGeometry(0.115, 14, 10),
        new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x2563eb, emissiveIntensity: 0.22, roughness: 0.45 })
      );
      handle.position.copy(sourceToVector(point, radius * 1.105));
      handle.userData = { type: 'control', course, section, point };
      rememberOpacity(handle);
      handles.add(handle);
    });

    (section.knowledge || []).forEach((point) => {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.11, 14, 10),
        new THREE.MeshStandardMaterial({ color: colorToThree(course.color, -0.3) })
      );
      dot.position.copy(sourceToVector(point, radius * 1.12));
      dot.userData = { type: 'knowledge', course, section, point };
      rememberOpacity(dot);
      knowledge.add(dot);

      const label = makeSurfaceLabel(point.label, '#0f172a', 0.35);
      placeSurfaceObject(label, sourceToVector(point), 1.14);
      label.userData = { type: 'knowledge', course, section, point };
      rememberOpacity(label);
      knowledgeLabels.add(label);
    });
  });

  const title = makeBillboardLabel(course.title, '#101820', 1.16);
  title.position.set(0, radius * 1.72, 0);
  title.userData = { type: 'course', course };
  group.add(title);

  courseGroups.push({
    course,
    group,
    base,
    outer,
    chapters,
    chapterLabels,
    sections,
    sectionLabels,
    knowledge,
    knowledgeLabels,
    handles,
    title
  });
}

data.courses.forEach(buildCourse);

function setLod(level) {
  lodLabel.textContent = `LOD ${level}`;
  currentLodLevel = level;
}

function updateLayerBlend(dist) {
  const titleAlpha = smoothstep(74, 92, dist);
  const outerAlpha = smoothstep(26, 36, dist);
  const chapterAlpha = (1 - smoothstep(78, 94, dist)) * smoothstep(46, 58, dist);
  const sectionAlpha = 1 - smoothstep(44, 54, dist);
  const sectionLabelAlpha = 1 - smoothstep(24, 42, dist);
  const knowledgeAlpha = 1 - smoothstep(22, 32, dist);
  const handleAlpha = editMode && currentLodLevel < 3 ? 0.96 : 0;
  courseGroups.forEach((item) => {
    setLayerOpacity(item.outer, outerAlpha);
    setLayerOpacity(item.chapters, chapterAlpha);
    setLayerOpacity(item.chapterLabels, chapterAlpha);
    setLayerOpacity(item.sections, sectionAlpha);
    setLayerOpacity(item.sectionLabels, Math.max(sectionAlpha * 0.42, sectionLabelAlpha));
    setLayerOpacity(item.knowledge, knowledgeAlpha);
    setLayerOpacity(item.knowledgeLabels, knowledgeAlpha);
    setLayerOpacity(item.handles, handleAlpha);
    setLayerOpacity(item.title, titleAlpha);
  });
}

function updateLod() {
  const dist = camera.position.distanceTo(controls.target);
  const level = dist > 82 ? 0 : dist > 43 ? 1 : dist > 25 ? 2 : 3;
  setLod(level);
  updateLayerBlend(dist);
}

function focusCourse(courseId) {
  const previousCourseId = selectedCourse?.course.id || null;
  selectedCourse = courseGroups.find((item) => item.course.id === courseId) || null;
  const changedFocus = previousCourseId !== (selectedCourse?.course.id || null);
  courseGroups.forEach((item, index) => {
    item.group.visible = !selectedCourse || item === selectedCourse;
    item.group.userData.target = selectedCourse
      ? new THREE.Vector3(item === selectedCourse ? 0 : (index < 2 ? -82 : 82), 0, selectedCourse && item !== selectedCourse ? -34 : 0)
      : overviewPosition(index);
  });
  if (selectedCourse) {
    courseTitle.textContent = selectedCourse.course.title;
    if (changedFocus) {
      controls.target.set(0, 0, 0);
      camera.position.set(0, 7, focusDistance);
    }
  } else {
    courseTitle.textContent = '四个教材';
    if (changedFocus) {
      controls.target.set(0, 0, 0);
      camera.position.set(0, 10, currentOverviewDistance());
    }
  }
  renderTabs();
}

function renderTabs() {
  tabsEl.innerHTML = '';
  data.courses.forEach((course) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = course.title;
    button.className = selectedCourse?.course.id === course.id ? 'active' : '';
    button.addEventListener('click', () => focusCourse(course.id));
    tabsEl.appendChild(button);
  });
}

function selectNode(object) {
  const item = object?.userData;
  if (!item?.type) return;
  selectedNode = { ...item, object };
  selectionText.textContent = describeSelection(item);
  const editablePoint = item.point;
  if (editablePoint) {
    phiInput.value = editablePoint.phi || 0;
    psiInput.value = editablePoint.psi || 0;
  }
}

function syncPointObjects(course, point) {
  const group = courseGroups.find((item) => item.course === course);
  if (!group || !point) return;
  group.outer.children.forEach((child) => {
    if (child.userData?.point === point) child.position.copy(sourceToVector(point, radius * 1.095));
  });
  group.knowledge.children.forEach((child) => {
    if (child.userData?.point === point) child.position.copy(sourceToVector(point, radius * 1.12));
  });
  group.knowledgeLabels.children.forEach((child) => {
    if (child.userData?.point === point) placeSurfaceObject(child, sourceToVector(point), 1.14);
  });
  group.handles.children.forEach((child) => {
    if (child.userData?.point === point) child.position.copy(sourceToVector(point, radius * 1.105));
  });
}

function updateSelectedPoint() {
  if (!selectedNode?.point) return;
  const local = sphericalToVector(Number(phiInput.value), Number(psiInput.value), radius);
  vectorToSource(local, selectedNode.point);
  syncPointObjects(selectedNode.course, selectedNode.point);
  if (selectedNode.type === 'control') rebuildSectionVisual(selectedNode.course, selectedNode.section);
}

function setPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
}

function raycastScene(event) {
  setPointer(event);
  const visibleGroups = courseGroups.filter((item) => item.group.visible);
  const targets = visibleGroups.flatMap((item) => [
    item.base,
    item.title,
    ...item.outer.children,
    ...item.chapters.children,
    ...item.chapterLabels.children,
    ...item.sections.children,
    ...item.sectionLabels.children,
    ...item.knowledge.children,
    ...item.knowledgeLabels.children,
    ...item.handles.children
  ]);
  return raycaster.intersectObjects(targets, false)[0];
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  const hit = raycastScene(event);
  if (!hit) return;
  const item = hit.object.userData;
  if (item?.course) {
    if (!selectedCourse || selectedCourse.course.id !== item.course.id) {
      focusCourse(item.course.id);
    }
    selectNode(hit.object);
  }
  if (editMode && item?.point) {
    draggingPoint = { course: item.course, point: item.point, section: item.section, type: item.type };
    controls.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
  }
});

renderer.domElement.addEventListener('pointermove', (event) => {
  if (!draggingPoint) return;
  setPointer(event);
  const group = courseGroups.find((item) => item.course === draggingPoint.course);
  if (!group) return;
  const center = new THREE.Vector3();
  group.group.getWorldPosition(center);
  const hit = new THREE.Vector3();
  const ok = raycaster.ray.intersectSphere(new THREE.Sphere(center, radius * 1.12), hit);
  if (!ok) return;
  const local = group.group.worldToLocal(hit.clone()).normalize().multiplyScalar(radius);
  vectorToSource(local, draggingPoint.point);
  phiInput.value = draggingPoint.point.phi;
  psiInput.value = draggingPoint.point.psi;
  syncPointObjects(draggingPoint.course, draggingPoint.point);
  if (draggingPoint.type === 'control') rebuildSectionVisual(draggingPoint.course, draggingPoint.section);
});

renderer.domElement.addEventListener('pointerup', (event) => {
  if (draggingPoint) {
    draggingPoint = null;
    controls.enabled = true;
    if (renderer.domElement.hasPointerCapture(event.pointerId)) {
      renderer.domElement.releasePointerCapture(event.pointerId);
    }
  }
});

overviewBtn.addEventListener('click', () => focusCourse(null));
editBtn.addEventListener('click', () => {
  editMode = !editMode;
  editBtn.classList.toggle('active', editMode);
  editorEl.classList.toggle('hidden', !editMode);
});
phiInput.addEventListener('input', updateSelectedPoint);
psiInput.addEventListener('input', updateSelectedPoint);
document.querySelectorAll('[data-nudge]').forEach((button) => {
  button.addEventListener('click', () => {
    if (!selectedNode?.point) return;
    const step = 0.04;
    if (button.dataset.nudge === 'left') phiInput.value = Number(phiInput.value) - step;
    if (button.dataset.nudge === 'right') phiInput.value = Number(phiInput.value) + step;
    if (button.dataset.nudge === 'up') psiInput.value = Number(psiInput.value) + step;
    if (button.dataset.nudge === 'down') psiInput.value = Number(psiInput.value) - step;
    updateSelectedPoint();
  });
});

function animate() {
  requestAnimationFrame(animate);
  courseGroups.forEach((item) => {
    const target = item.group.userData.target || item.group.position;
    item.group.position.lerp(target, 0.08);
  });
  controls.update();
  updateLod();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (!selectedCourse) {
    courseGroups.forEach((item, index) => {
      item.group.userData.target = overviewPosition(index);
    });
    camera.position.set(0, 10, currentOverviewDistance());
  }
});

camera.position.set(0, 10, currentOverviewDistance());
renderTabs();
const initialCourseId = new URLSearchParams(window.location.search).get('course');
if (initialCourseId && data.courses.some((course) => course.id === initialCourseId)) {
  focusCourse(initialCourseId);
} else {
  setLod(0);
}
animate();
