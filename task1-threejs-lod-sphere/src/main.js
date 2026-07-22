import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import data from './data/course-data.json';
import {
  ConfigValidationError,
  createEmptyCourse,
  createEntityId,
  exportConfig,
  materializeGeneratedKnowledge,
  normalizeConfig,
  pointFromAngles,
  replaceConfig,
  serializeConfig
} from './config-model.js';
import {
  createSemanticFields,
  createSemanticMaterial,
  disposeSemanticFields,
  findRegionAtDirection,
  invalidateCourseLayouts,
  prepareCourse,
  regionLayout,
  sampleClosedBoundary,
  slerpUnit,
  slerpWithinBoundary,
  smoothstep,
  sphericalContains,
  sourceToVector,
  updateMaterialFields,
  vectorToSource
} from './spherical-field.js';
import {
  constrainKnowledgeDirection,
  findClosestSharedBoundary,
  sharedBoundaryNeighbors
} from './editor-constraints.js';
import './styles.css';

const sceneEl = document.querySelector('#scene');
const tabsEl = document.querySelector('#courseTabs');
const outlineEl = document.querySelector('#outline');
const lodLabel = document.querySelector('#lodLabel');
const courseTitle = document.querySelector('#courseTitle');
const selectionText = document.querySelector('#selectionText');
const overviewBtn = document.querySelector('#overviewBtn');
const editBtn = document.querySelector('#editBtn');
const addVertexBtn = document.querySelector('#addVertexBtn');
const addCourseBtn = document.querySelector('#addCourseBtn');
const importBtn = document.querySelector('#importBtn');
const exportBtn = document.querySelector('#exportBtn');
const undoBtn = document.querySelector('#undoBtn');
const redoBtn = document.querySelector('#redoBtn');
const fileInput = document.querySelector('#fileInput');
const dirtyBadge = document.querySelector('#dirtyBadge');
const configSummary = document.querySelector('#configSummary');
const inspectorEl = document.querySelector('#inspector');
const selectionActionsEl = document.querySelector('#selectionActions');
const editorEl = document.querySelector('#editor');
const phiInput = document.querySelector('#phiInput');
const psiInput = document.querySelector('#psiInput');
const placementBar = document.querySelector('#placementBar');
const placementTitle = document.querySelector('#placementTitle');
const placementHint = document.querySelector('#placementHint');
const draftNameField = document.querySelector('#draftNameField');
const draftNameInput = document.querySelector('#draftNameInput');
const finishPlacementBtn = document.querySelector('#finishPlacementBtn');
const cancelPlacementBtn = document.querySelector('#cancelPlacementBtn');
const toastEl = document.querySelector('#toast');
const outlineSearch = document.querySelector('#outlineSearch');
const lodHint = document.querySelector('#lodHint');
const lodProgress = document.querySelector('#lodProgress');
const sceneStatus = document.querySelector('#sceneStatus');
const libraryPanel = document.querySelector('#libraryPanel');
const inspectorPanel = document.querySelector('#panel');
const hideLibraryBtn = document.querySelector('#hideLibraryBtn');
const hideInspectorBtn = document.querySelector('#hideInspectorBtn');
const showLibraryBtn = document.querySelector('#showLibraryBtn');
const showInspectorBtn = document.querySelector('#showInspectorBtn');
const helpBtn = document.querySelector('#helpBtn');
const helpDialog = document.querySelector('#helpDialog');

const initialConfig = normalizeConfig(data);
replaceConfig(data, initialConfig.config);

let radius = data.radius || 10;
const overviewDistance = 122;
const overviewSpacing = 26.5;
const focusDistance = 55;

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1200);
camera.position.set(0, 10, currentOverviewDistance());

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: true,
  powerPreference: 'high-performance'
});
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.02;
renderer.domElement.setAttribute('aria-label', '可旋转缩放的课程球面知识图谱');
sceneEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 14;
controls.maxDistance = 170;

scene.add(new THREE.HemisphereLight(0xffffff, 0x7f91aa, 1.45));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.15);
keyLight.position.set(28, 36, 30);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xc6f2ff, 0.9);
rimLight.position.set(-30, 16, -24);
scene.add(rimLight);

// 轻量的程序化星尘让课程球更像一个可探索的知识空间，且不依赖外部图片资源。
function createStarField() {
  let seed = 42689;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };
  const positions = [];
  for (let index = 0; index < 520; index += 1) {
    const distance = 170 + random() * 310;
    const theta = random() * Math.PI * 2;
    const phi = Math.acos(2 * random() - 1);
    positions.push(
      distance * Math.sin(phi) * Math.cos(theta),
      distance * Math.cos(phi),
      distance * Math.sin(phi) * Math.sin(theta)
    );
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0x93b9e8,
    size: 0.42,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
    toneMapped: false
  });
  const stars = new THREE.Points(geometry, material);
  stars.renderOrder = -10;
  scene.add(stars);
  return stars;
}

const starField = createStarField();

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
let rebuildTimer = null;
let placementMode = null;
let toastTimer = null;
let sliderBefore = null;
let sliderLimited = false;
let savedSnapshot = serializeConfig(data, 0);
const undoStack = [];
const redoStack = [];
const expandedIds = new Set();

const outerGeometry = new THREE.SphereGeometry(0.17, 18, 14);
const handleGeometry = new THREE.SphereGeometry(0.22, 18, 14);
const knowledgeGeometry = new THREE.SphereGeometry(0.17, 18, 14);
const knowledgeLabelCache = new Map();
const surfaceLabelColor = '#ffffff';

function isMobileView() {
  return window.innerWidth < 720;
}

function currentOverviewDistance() {
  const columns = isMobileView()
    ? Math.min(2, Math.max(1, data.courses.length))
    : Math.ceil(Math.sqrt(Math.max(1, data.courses.length)));
  const rows = Math.ceil(Math.max(1, data.courses.length) / columns);
  const layoutScale = Math.max(columns, rows);
  return Math.max(isMobileView() ? 150 : overviewDistance, 70 + layoutScale * 25);
}

function overviewPosition(index) {
  const count = Math.max(1, data.courses.length);
  const columns = isMobileView() ? Math.min(2, count) : Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / columns);
  const column = index % columns;
  const row = Math.floor(index / columns);
  const spacing = isMobileView() ? 22 : overviewSpacing;
  return new THREE.Vector3(
    (column - (columns - 1) / 2) * spacing,
    ((rows - 1) / 2 - row) * spacing,
    0
  );
}

function wrapText(text, maxChars) {
  const clean = String(text).replace(/\s+/g, ' ').trim();
  const parts = clean.match(new RegExp(`.{1,${maxChars}}`, 'g')) || [clean];
  if (parts.length <= 3) return parts;
  return [parts[0], parts[1], `${parts.slice(2).join('').slice(0, maxChars - 1)}…`];
}

function makeTextTexture(text, color = '#17202a', maxChars = 10) {
  const lines = wrapText(text, maxChars);
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 32 + lines.length * 48;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = '700 32px "Microsoft YaHei", "Noto Sans SC", sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.lineJoin = 'round';
  context.lineWidth = color === '#ffffff' ? 6 : 4;
  context.strokeStyle = color === '#ffffff' ? 'rgba(6, 14, 28, 0.72)' : 'rgba(255, 255, 255, 0.78)';
  context.fillStyle = color;
  context.shadowColor = 'rgba(0, 0, 0, 0.28)';
  context.shadowBlur = 5;
  context.shadowOffsetY = 2;
  lines.forEach((line, index) => {
    const y = 16 + (index + 0.5) * 48;
    context.strokeText(line, canvas.width / 2, y);
    context.fillText(line, canvas.width / 2, y);
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
  return { texture, aspect: canvas.width / canvas.height };
}

function createCurvedGeometry(width, height) {
  const geometry = new THREE.PlaneGeometry(width, height, 14, 5);
  const positions = geometry.attributes.position;
  for (let index = 0; index < positions.count; index += 1) {
    const x = positions.getX(index);
    const y = positions.getY(index);
    const squaredDistance = Math.min(radius * radius * 0.92, x * x + y * y);
    positions.setZ(index, Math.sqrt(radius * radius - squaredDistance) - radius);
  }
  positions.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function surfaceQuaternion(normal) {
  const direction = normal.clone().normalize();
  const upReference = Math.abs(direction.y) > 0.94
    ? new THREE.Vector3(0, 0, 1)
    : new THREE.Vector3(0, 1, 0);
  const east = new THREE.Vector3().crossVectors(upReference, direction).normalize();
  const north = new THREE.Vector3().crossVectors(direction, east).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(east, north, direction));
}

function makeSurfaceLabel(text, color, layout, options = {}) {
  const maxChars = options.maxChars || 10;
  const { texture, aspect } = makeTextTexture(text, color, maxChars);
  const available = Math.max(0.9, 2 * radius * Math.sin(layout.clearance) * 0.88);
  const targetWidth = options.targetWidth || 5.4;
  let width = targetWidth;
  let height = width / aspect;
  const maxHeight = Math.max(0.45, available * 0.62);
  const fitScale = Math.min(1, available / width, maxHeight / height);
  // 标签使用同一目标字号；极小区域最多缩小 20%。
  // 这样能保持整体字级统一，同时避免小区域文字无限缩小到不可读。
  const labelScale = Math.max(options.minScale || 0.8, fitScale);
  width *= labelScale;
  height *= labelScale;
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    alphaTest: 0.025,
    // 球面深度和标签网格非常接近，旋转时继续参与深度测试会产生闪烁。
    // 标签的前后面显隐由 updateSurfaceLabelVisibility 统一处理。
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false
  });
  material.userData.baseOpacity = 1;
  const mesh = new THREE.Mesh(createCurvedGeometry(width, height), material);
  const radialOffset = options.radialOffset || 0.22;
  mesh.position.copy(layout.anchor).multiplyScalar(radius + radialOffset);
  mesh.quaternion.copy(surfaceQuaternion(layout.anchor));
  mesh.renderOrder = options.renderOrder || 5;
  mesh.userData = {
    isSurfaceLabel: true,
    anchorLocal: mesh.position.clone(),
    labelSize: { width, height },
    labelPriority: options.priority || 1
  };
  return mesh;
}

function makeBillboardLabel(text, color = '#17202a', scale = 1, depthTest = false) {
  const { texture, aspect } = makeTextTexture(text, color, 12);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest, depthWrite: false });
  material.userData.baseOpacity = 1;
  const sprite = new THREE.Sprite(material);
  const height = 1.35 * scale;
  sprite.scale.set(height * aspect, height, 1);
  return sprite;
}

function makeKnowledgeLabel(text) {
  if (!knowledgeLabelCache.has(text)) {
    const { texture, aspect } = makeTextTexture(text, surfaceLabelColor, 3);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      alphaTest: 0.025,
      depthTest: false,
      depthWrite: false,
      toneMapped: false
    });
    material.userData.baseOpacity = 1;
    material.userData.sharedMaterial = true;
    knowledgeLabelCache.set(text, { material, aspect });
  }
  const cached = knowledgeLabelCache.get(text);
  const sprite = new THREE.Sprite(cached.material);
  const height = 0.46;
  sprite.scale.set(height * cached.aspect, height, 1);
  return sprite;
}

function makeShadowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(128, 128, 12, 128, 128, 118);
  gradient.addColorStop(0, 'rgba(26, 39, 58, 0.34)');
  gradient.addColorStop(0.55, 'rgba(26, 39, 58, 0.16)');
  gradient.addColorStop(1, 'rgba(26, 39, 58, 0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(canvas);
}

const shadowTexture = makeShadowTexture();

function clearDynamicGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    if (!child.userData?.sharedGeometry) child.geometry?.dispose();
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    materials.forEach((material) => {
      if (material.userData?.sharedMaterial) return;
      material.map?.dispose();
      material.dispose();
    });
  }
}

function setLayerOpacity(group, opacity) {
  group.visible = opacity > 0.015;
  group.traverse((child) => {
    const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
    materials.forEach((material) => {
      if (material.userData.baseOpacity === undefined) material.userData.baseOpacity = material.opacity ?? 1;
      material.opacity = material.userData.baseOpacity * opacity;
      material.transparent = true;
      material.depthWrite = false;
    });
  });
}

function buildVertexUsage(course) {
  const usage = new Map(course.vertices.map((vertex) => [vertex.id, new Set()]));
  [...course.chapters, ...course.sections].forEach((region) => {
    region.vertexIds.forEach((vertexId) => usage.get(vertexId)?.add(region));
  });
  course.__vertexUsage = usage;
  course.__boundaryVertices = course.vertices.filter((vertex) => usage.get(vertex.id)?.size);
}

function buildOuterPoints(courseGroup) {
  const { course, outer } = courseGroup;
  const boundaryVertices = course.__boundaryVertices || [];
  const stride = Math.max(1, Math.ceil(boundaryVertices.length / 8));
  boundaryVertices.forEach((point, index) => {
    if (index % stride !== 0 && index !== boundaryVertices.length - 1) return;
    const material = new THREE.MeshStandardMaterial({
      color: 0xf8fbff,
      emissive: new THREE.Color(course.color),
      emissiveIntensity: index % 2 ? 0.18 : 0.28,
      roughness: 0.46,
      metalness: 0.02,
      transparent: true
    });
    const dot = new THREE.Mesh(outerGeometry, material);
    dot.userData = { type: 'point', course, point, sharedGeometry: true };
    dot.position.copy(sourceToVector(point, radius + 0.82));
    outer.add(dot);
  });
}

function buildHandles(courseGroup) {
  const { course, handles } = courseGroup;
  course.vertices.forEach((point) => {
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0x1677ff,
      emissiveIntensity: 0.45,
      roughness: 0.34,
      transparent: true
    });
    const handle = new THREE.Mesh(handleGeometry, material);
    handle.userData = { type: 'control', course, point, sharedGeometry: true };
    handle.position.copy(sourceToVector(point, radius + 0.46));
    handle.renderOrder = 12;
    handles.add(handle);
  });
}

function ensureKnowledge(section, course) {
  if (!Array.isArray(section.knowledge)) section.knowledge = [];
  // 内置课程的源表只有章/节边界，没有独立的知识点记录。沿用展示版行为，
  // 为这类小节生成三个仅运行时存在的球面知识点；序列化时 generated 点会被过滤，
  // 因而不会污染用户导出的课程配置。
  if (!section.knowledge.length) {
    section.knowledge = [0, 1, 2].map((index) => ({
      id: `${section.id}-knowledge-${index + 1}`,
      label: `K${index + 1}`,
      generated: true
    }));
  }
  const layout = regionLayout(course, section);
  const boundary = sampleClosedBoundary(course, section, 6, 1);
  section.knowledge.forEach((point, index) => {
    if (point.manual) return;
    const target = boundary[Math.floor(((index + 0.65) / section.knowledge.length) * boundary.length) % boundary.length];
    const direction = target
      ? slerpWithinBoundary(layout.anchor, target, index === 0 ? 0.18 : 0.34, boundary)
      : layout.anchor;
    vectorToSource(direction.multiplyScalar(radius), point, radius);
  });
}

function buildLabelsAndKnowledge(courseGroup) {
  const {
    course, chapterLabels, sectionLabels, knowledge, knowledgeLabels, knowledgeMaterial
  } = courseGroup;
  clearDynamicGroup(chapterLabels);
  clearDynamicGroup(sectionLabels);
  clearDynamicGroup(knowledge);
  clearDynamicGroup(knowledgeLabels);

  course.chapters.forEach((chapter) => {
    const layout = regionLayout(course, chapter);
    const label = makeSurfaceLabel(chapter.title, surfaceLabelColor, layout, {
      maxChars: 8,
      targetWidth: 5.4,
      minScale: 0.8,
      radialOffset: 0.25,
      priority: 3,
      renderOrder: 7
    });
    label.userData = { ...label.userData, type: 'chapter', course, chapter };
    chapterLabels.add(label);
  });

  course.sections.forEach((section) => {
    const layout = regionLayout(course, section);
    const label = makeSurfaceLabel(section.title, surfaceLabelColor, layout, {
      maxChars: 9,
      targetWidth: 5.4,
      minScale: 0.8,
      radialOffset: 0.27,
      priority: 2,
      renderOrder: 8
    });
    label.userData = { ...label.userData, type: 'section', course, section };
    sectionLabels.add(label);

    ensureKnowledge(section, course);
    section.knowledge.forEach((point) => {
      const dot = new THREE.Mesh(knowledgeGeometry, knowledgeMaterial);
      dot.position.copy(sourceToVector(point, radius + 0.34));
      dot.userData = { type: 'knowledge', course, section, point, sharedGeometry: true };
      knowledge.add(dot);

      const pointLabel = makeKnowledgeLabel(point.label);
      pointLabel.position.copy(sourceToVector(point, radius + 0.48));
      pointLabel.userData = {
        type: 'knowledge',
        course,
        section,
        point,
        isSurfaceLabel: true,
        anchorLocal: pointLabel.position.clone(),
        labelSize: { width: 0.86, height: 0.38 },
        labelPriority: 1,
        sharedGeometry: true
      };
      knowledgeLabels.add(pointLabel);
    });
  });
}

function buildPreviewGeometry(course) {
  const positions = [];
  const boundaryEdges = new Map();
  [...course.chapters, ...course.sections].forEach((region) => {
    const ids = region.vertexIds || [];
    ids.forEach((startId, index) => {
      const endId = ids[(index + 1) % ids.length];
      if (!startId || !endId || startId === endId) return;
      const key = [startId, endId].sort().join('|');
      if (!boundaryEdges.has(key)) boundaryEdges.set(key, { a: startId, b: endId });
    });
  });
  boundaryEdges.forEach((edge) => {
    const startPoint = course.__vertexMap.get(edge.a);
    const endPoint = course.__vertexMap.get(edge.b);
    if (!startPoint || !endPoint) return;
    const start = sourceToVector(startPoint, 1);
    const end = sourceToVector(endPoint, 1);
    const steps = 8;
    for (let step = 0; step < steps; step += 1) {
      const a = slerpUnit(start, end, step / steps).multiplyScalar(radius + 0.31);
      const b = slerpUnit(start, end, (step + 1) / steps).multiplyScalar(radius + 0.31);
      positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

function updatePreview(courseGroup) {
  const oldGeometry = courseGroup.preview.geometry;
  courseGroup.preview.geometry = buildPreviewGeometry(courseGroup.course);
  oldGeometry?.dispose();
}

function updateFocusDirection(courseGroup) {
  const layouts = courseGroup.course.chapters.map((chapter) => regionLayout(courseGroup.course, chapter));
  const largest = layouts.reduce((best, layout) => (
    !best || layout.clearance > best.clearance ? layout : best
  ), null);
  courseGroup.focusDirection = largest?.anchor.clone().normalize() || new THREE.Vector3(0, 0, 1);
}

function buildCourse(course, index) {
  invalidateCourseLayouts(course);
  prepareCourse(course);
  buildVertexUsage(course);

  const group = new THREE.Group();
  group.position.copy(overviewPosition(index));
  group.userData.target = group.position.clone();
  root.add(group);

  const fields = createSemanticFields(course);
  const material = createSemanticMaterial(fields, course.color);
  const base = new THREE.Mesh(new THREE.SphereGeometry(radius, 192, 128), material);
  base.userData = { type: 'course', course };
  group.add(base);

  const shadowMaterial = new THREE.MeshBasicMaterial({
    map: shadowTexture,
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
    toneMapped: false
  });
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(19, 19), shadowMaterial);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = -radius - 0.48;
  shadow.renderOrder = -1;
  group.add(shadow);

  const outer = new THREE.Group();
  const chapterLabels = new THREE.Group();
  const sectionLabels = new THREE.Group();
  const knowledge = new THREE.Group();
  const knowledgeLabels = new THREE.Group();
  const handles = new THREE.Group();
  const draft = new THREE.Group();
  group.add(outer, chapterLabels, sectionLabels, knowledge, knowledgeLabels, handles, draft);

  const previewMaterial = new THREE.LineBasicMaterial({
    color: 0x0b76d1,
    transparent: true,
    opacity: 0.84,
    depthTest: true,
    depthWrite: false
  });
  previewMaterial.userData.baseOpacity = 0.84;
  const preview = new THREE.LineSegments(buildPreviewGeometry(course), previewMaterial);
  preview.renderOrder = 10;
  group.add(preview);

  const knowledgeMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color(course.color).lerp(new THREE.Color(0x07111f), 0.44),
    emissive: new THREE.Color(course.color),
    emissiveIntensity: 0.14,
    roughness: 0.45,
    transparent: true
  });
  knowledgeMaterial.userData.sharedMaterial = true;
  knowledgeMaterial.userData.baseOpacity = 1;

  const title = makeBillboardLabel(course.title, '#ffffff', 1.55, false);
  title.position.set(0, radius * 1.52, 0);
  title.userData = { type: 'course', course };
  group.add(title);

  const courseGroup = {
    course,
    group,
    base,
    material,
    fields,
    shadow,
    outer,
    chapterLabels,
    sectionLabels,
    knowledge,
    knowledgeLabels,
    handles,
    draft,
    preview,
    knowledgeMaterial,
    title
  };
  buildOuterPoints(courseGroup);
  buildHandles(courseGroup);
  buildLabelsAndKnowledge(courseGroup);
  updateFocusDirection(courseGroup);
  courseGroups.push(courseGroup);
}

data.courses.forEach(buildCourse);

function showToast(message, duration = 2600) {
  clearTimeout(toastTimer);
  toastEl.textContent = message;
  toastEl.classList.remove('hidden');
  toastTimer = setTimeout(() => toastEl.classList.add('hidden'), duration);
}

function snapshotConfig() {
  return serializeConfig(data, 0);
}

function updateHistoryUI() {
  undoBtn.disabled = undoStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
  const dirty = snapshotConfig() !== savedSnapshot;
  dirtyBadge.textContent = dirty ? '未导出' : '已保存';
  dirtyBadge.classList.toggle('dirty', dirty);
}

function commitHistory(before, message) {
  const after = snapshotConfig();
  if (before === after) return false;
  undoStack.push(before);
  if (undoStack.length > 80) undoStack.shift();
  redoStack.length = 0;
  updateHistoryUI();
  if (message) showToast(message);
  return true;
}

function selectionLocator(item = selectedNode) {
  if (!item?.course) return null;
  const base = { type: item.type, courseId: item.course.id };
  if (item.chapter) base.id = item.chapter.id;
  if (item.section) {
    if (item.type === 'section') base.id = item.section.id;
    else base.sectionId = item.section.id;
  }
  if (item.point) {
    base.id = item.point.id;
    if (item.type === 'control' || item.type === 'point') base.type = 'vertex';
  }
  return base;
}

function resolveLocator(locator) {
  if (!locator) return null;
  const course = data.courses.find((item) => item.id === locator.courseId);
  if (!course) return null;
  if (locator.type === 'course') return { type: 'course', course };
  if (locator.type === 'chapter') {
    const chapter = course.chapters.find((item) => item.id === locator.id);
    return chapter ? { type: 'chapter', course, chapter } : null;
  }
  if (locator.type === 'section') {
    const section = course.sections.find((item) => item.id === locator.id);
    return section ? { type: 'section', course, section } : null;
  }
  if (locator.type === 'knowledge') {
    const section = course.sections.find((item) => item.id === locator.sectionId);
    const point = section?.knowledge.find((item) => item.id === locator.id);
    return point ? { type: 'knowledge', course, section, point } : null;
  }
  if (locator.type === 'vertex') {
    const point = course.vertices.find((item) => item.id === locator.id);
    return point ? { type: 'control', course, point } : null;
  }
  return null;
}

function disposeCourseGroup(courseGroup) {
  disposeSemanticFields(courseGroup.fields);
  courseGroup.group.traverse((child) => {
    if (child.geometry && !child.userData?.sharedGeometry) child.geometry.dispose();
    const materials = child.material
      ? (Array.isArray(child.material) ? child.material : [child.material])
      : [];
    materials.forEach((material) => {
      if (material.userData?.sharedMaterial) return;
      if (material.map && material.map !== shadowTexture) material.map.dispose();
      material.dispose();
    });
  });
  courseGroup.knowledgeMaterial.dispose();
  root.remove(courseGroup.group);
}

function clearKnowledgeLabelCache() {
  knowledgeLabelCache.forEach(({ material }) => {
    material.map?.dispose();
    material.dispose();
  });
  knowledgeLabelCache.clear();
}

function rebuildAllCourses(locator = selectionLocator(), focusId = selectedCourse?.course.id || null) {
  clearTimeout(rebuildTimer);
  rebuildTimer = null;
  const cameraPosition = camera.position.clone();
  const target = controls.target.clone();
  courseGroups.forEach(disposeCourseGroup);
  courseGroups.length = 0;
  clearKnowledgeLabelCache();
  selectedCourse = null;
  selectedNode = null;
  radius = Number(data.radius) || 10;
  data.courses.forEach(buildCourse);
  focusCourse(focusId && data.courses.some((course) => course.id === focusId) ? focusId : null);
  camera.position.copy(cameraPosition);
  controls.target.copy(target);
  const resolved = resolveLocator(locator);
  if (resolved) selectNode(resolved);
  else renderEditorUI();
  renderDraft();
}

function rebuildSingleCourse(courseId, locator = selectionLocator()) {
  const course = data.courses.find((item) => item.id === courseId);
  if (!course) {
    rebuildAllCourses(locator, null);
    return;
  }
  const cameraPosition = camera.position.clone();
  const target = controls.target.clone();
  const oldIndex = courseGroups.findIndex((item) => item.course.id === courseId);
  if (oldIndex >= 0) {
    disposeCourseGroup(courseGroups[oldIndex]);
    courseGroups.splice(oldIndex, 1);
  }
  const dataIndex = data.courses.indexOf(course);
  selectedCourse = null;
  selectedNode = null;
  buildCourse(course, dataIndex);
  const created = courseGroups.pop();
  courseGroups.splice(Math.min(dataIndex, courseGroups.length), 0, created);
  focusCourse(courseId);
  camera.position.copy(cameraPosition);
  controls.target.copy(target);
  const resolved = resolveLocator(locator);
  if (resolved) selectNode(resolved);
  else renderEditorUI();
  renderDraft();
}

function applyHistorySnapshot(snapshot, direction) {
  const current = snapshotConfig();
  const parsed = JSON.parse(snapshot);
  replaceConfig(data, parsed);
  cancelPlacement(false);
  if (direction === 'undo') redoStack.push(current);
  else undoStack.push(current);
  rebuildAllCourses(null, null);
  updateHistoryUI();
  showToast(direction === 'undo' ? '已撤销' : '已重做');
}

function undo() {
  if (!undoStack.length) return;
  applyHistorySnapshot(undoStack.pop(), 'undo');
}

function redo() {
  if (!redoStack.length) return;
  applyHistorySnapshot(redoStack.pop(), 'redo');
}

function rebuildCourseVisuals(courseGroup) {
  invalidateCourseLayouts(courseGroup.course);
  const nextFields = createSemanticFields(courseGroup.course);
  updateMaterialFields(courseGroup.material, nextFields);
  disposeSemanticFields(courseGroup.fields);
  courseGroup.fields = nextFields;
  buildLabelsAndKnowledge(courseGroup);
  updateFocusDirection(courseGroup);
  updatePreview(courseGroup);
}

function queueCourseRebuild(courseGroup, delay = 180) {
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(() => {
    rebuildTimer = null;
    rebuildCourseVisuals(courseGroup);
  }, delay);
}

function setLod(level) {
  if (currentLodLevel === level && lodLabel.textContent === `LOD ${level}`) return;
  currentLodLevel = level;
  lodLabel.textContent = `LOD ${level}`;
  const levelNames = ['课程总览', '章节脉络', '小节结构', '知识节点'];
  lodHint.textContent = levelNames[level];
  lodProgress.querySelectorAll('span').forEach((dot, index) => {
    dot.classList.toggle('active', index === level);
    dot.classList.toggle('passed', index < level);
  });
  sceneStatus.textContent = selectedCourse
    ? `${selectedCourse.course.title} · ${levelNames[level]}`
    : '课程宇宙总览';
}

function updateLayerBlend(distance) {
  const regionReveal = 1 - smoothstep(78, 92, distance);
  const sectionBlend = 1 - smoothstep(43, 56, distance);
  const knowledgeAlpha = 1 - smoothstep(24, 31, distance);
  // 微调控制球与部分抽样外层点位于同一径向位置，二者同时显示会形成重影。
  const outerAlpha = editMode ? 0 : smoothstep(24, 31, distance);
  const titleAlpha = smoothstep(74, 94, distance);
  const chapterLabelAlpha = regionReveal * (1 - sectionBlend);
  const sectionLabelAlpha = regionReveal * sectionBlend;
  const handleAlpha = editMode ? 0.96 : 0;

  courseGroups.forEach((item) => {
    item.material.uniforms.uRegionReveal.value = regionReveal;
    item.material.uniforms.uSectionBlend.value = sectionBlend;
    setLayerOpacity(item.outer, outerAlpha);
    setLayerOpacity(item.chapterLabels, chapterLabelAlpha);
    setLayerOpacity(item.sectionLabels, sectionLabelAlpha);
    setLayerOpacity(item.knowledge, knowledgeAlpha);
    setLayerOpacity(item.knowledgeLabels, knowledgeAlpha);
    setLayerOpacity(item.handles, handleAlpha);
    setLayerOpacity(item.preview, handleAlpha);
    setLayerOpacity(item.title, titleAlpha);
  });
}

function updateLod() {
  const distance = camera.position.distanceTo(controls.target);
  const level = distance > 82 ? 0 : distance > 48 ? 1 : distance > 28 ? 2 : 3;
  setLod(level);
  updateLayerBlend(distance);
}

function isSurfaceLabelVisible(label, courseGroup) {
  const material = Array.isArray(label.material) ? label.material[0] : label.material;
  if (!label.userData?.isSurfaceLabel || !material || material.opacity < 0.12) return false;
  const localAnchor = label.userData.anchorLocal;
  if (!localAnchor) return false;
  const world = courseGroup.group.localToWorld(localAnchor.clone());
  const center = new THREE.Vector3();
  courseGroup.group.getWorldPosition(center);
  const normal = world.clone().sub(center).normalize();
  const view = camera.position.clone().sub(world).normalize();
  // 使用显隐滞回区间，避免标签在球体轮廓附近因浮点误差反复跳变。
  const facingThreshold = label.visible ? -0.06 : 0.06;
  if (normal.dot(view) <= facingThreshold) return false;
  const projected = world.clone().project(camera);
  return Math.abs(projected.x) <= 1.08
    && Math.abs(projected.y) <= 1.08
    && projected.z >= -1
    && projected.z <= 1;
}

function updateSurfaceLabelVisibility() {
  courseGroups.forEach((courseGroup) => {
    if (!courseGroup.group.visible) return;
    courseGroup.group.updateMatrixWorld();
    [
      ...courseGroup.chapterLabels.children,
      ...courseGroup.sectionLabels.children,
      ...courseGroup.knowledgeLabels.children
    ].forEach((label) => {
      label.visible = isSurfaceLabelVisible(label, courseGroup);
    });
  });
}

function renderTabs() {
  tabsEl.innerHTML = '';
  data.courses.forEach((course) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = course.title;
    button.className = selectedCourse?.course.id === course.id ? 'active' : '';
    button.setAttribute('aria-pressed', selectedCourse?.course.id === course.id ? 'true' : 'false');
    button.addEventListener('click', () => focusCourse(course.id));
    tabsEl.appendChild(button);
  });
}

function focusCourse(courseId) {
  const previousId = selectedCourse?.course.id || null;
  selectedCourse = courseGroups.find((item) => item.course.id === courseId) || null;
  const changed = previousId !== (selectedCourse?.course.id || null);
  courseGroups.forEach((item, index) => {
    item.group.visible = !selectedCourse || item === selectedCourse;
    item.group.userData.target = selectedCourse
      ? new THREE.Vector3(item === selectedCourse ? 0 : (index < 2 ? -82 : 82), 0, item === selectedCourse ? 0 : -34)
      : overviewPosition(index);
  });
  if (selectedCourse) {
    courseTitle.textContent = selectedCourse.course.title;
    sceneStatus.textContent = `${selectedCourse.course.title} · ${lodHint.textContent}`;
    if (changed) {
      const targetY = isMobileView() ? -3.2 : 0;
      controls.target.set(0, targetY, 0);
      const horizontal = selectedCourse.focusDirection.clone();
      horizontal.y = 0;
      if (horizontal.lengthSq() < 0.01) horizontal.set(0, 0, 1);
      horizontal.normalize().multiplyScalar(Math.sqrt(focusDistance * focusDistance - 49));
      camera.position.copy(horizontal);
      camera.position.y = targetY + 7;
    }
  } else {
    courseTitle.textContent = `${data.courses.length} 门课程`;
    selectionText.textContent = '未选择节点';
    sceneStatus.textContent = '课程宇宙总览';
    if (changed) {
      controls.target.set(0, 0, 0);
      camera.position.set(0, 10, currentOverviewDistance());
    }
  }
  renderTabs();
  renderOutline();
  updateConfigSummary();
}

function selectedOutlineKey() {
  const locator = selectionLocator();
  if (!locator) return '';
  return `${locator.type}:${locator.courseId}:${locator.sectionId || ''}:${locator.id || ''}`;
}

function outlineKey(type, courseId, id = '', sectionId = '') {
  return `${type}:${courseId}:${sectionId}:${id}`;
}

function makeOutlineButton(text, key, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'outline-button';
  button.textContent = text;
  button.title = text;
  button.classList.toggle('active', selectedOutlineKey() === key);
  button.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    onClick();
  });
  return button;
}

function createOutlineDetails(id, text, key, onSelect, forceOpen = false) {
  const details = document.createElement('details');
  details.open = forceOpen || expandedIds.has(id);
  details.addEventListener('toggle', () => {
    if (details.open) expandedIds.add(id);
    else expandedIds.delete(id);
  });
  const summary = document.createElement('summary');
  summary.appendChild(makeOutlineButton(text, key, onSelect));
  const children = document.createElement('div');
  children.className = 'outline-children';
  details.append(summary, children);
  return { details, children };
}

function renderOutline() {
  outlineEl.innerHTML = '';
  if (!data.courses.length) {
    const empty = document.createElement('p');
    empty.className = 'outline-empty';
    empty.textContent = '还没有课程。点击“＋课程”开始创建。';
    outlineEl.appendChild(empty);
    return;
  }

  data.courses.forEach((course) => {
    const courseSelected = selectedCourse?.course.id === course.id;
    const courseTree = createOutlineDetails(
      course.id,
      course.title,
      outlineKey('course', course.id),
      () => {
        focusCourse(course.id);
        selectNode({ type: 'course', course });
      },
      courseSelected
    );
    course.chapters.forEach((chapter) => {
      const chapterSelected = selectedNode?.chapter?.id === chapter.id
        || selectedNode?.section?.chapterId === chapter.id;
      const chapterTree = createOutlineDetails(
        chapter.id,
        chapter.title,
        outlineKey('chapter', course.id, chapter.id),
        () => {
          focusCourse(course.id);
          selectNode({ type: 'chapter', course, chapter });
        },
        chapterSelected
      );
      const sections = course.sections.filter((section) => section.chapterId === chapter.id);
      if (!sections.length) {
        const empty = document.createElement('p');
        empty.className = 'outline-empty';
        empty.textContent = '暂无小节';
        chapterTree.children.appendChild(empty);
      }
      sections.forEach((section) => {
        const sectionHasKnowledge = section.knowledge?.length > 0;
        const sectionSelected = selectedNode?.section?.id === section.id;
        if (sectionHasKnowledge) {
          const sectionTree = createOutlineDetails(
            section.id,
            section.title,
            outlineKey('section', course.id, section.id),
            () => {
              focusCourse(course.id);
              selectNode({ type: 'section', course, section });
            },
            sectionSelected
          );
          sectionTree.children.classList.add('outline-knowledge');
          section.knowledge.forEach((point) => {
            sectionTree.children.appendChild(makeOutlineButton(
              point.label,
              outlineKey('knowledge', course.id, point.id, section.id),
              () => {
                focusCourse(course.id);
                selectNode({ type: 'knowledge', course, section, point });
              }
            ));
          });
          chapterTree.children.appendChild(sectionTree.details);
        } else {
          chapterTree.children.appendChild(makeOutlineButton(
            section.title,
            outlineKey('section', course.id, section.id),
            () => {
              focusCourse(course.id);
              selectNode({ type: 'section', course, section });
            }
          ));
        }
      });
      courseTree.children.appendChild(chapterTree.details);
    });
    if (!course.chapters.length) {
      const empty = document.createElement('p');
      empty.className = 'outline-empty';
      empty.textContent = '暂无章节';
      courseTree.children.appendChild(empty);
    }
    outlineEl.appendChild(courseTree.details);
  });
  applyOutlineFilter(outlineSearch.value);
}

function applyOutlineFilter(value) {
  const query = String(value || '').trim().toLocaleLowerCase('zh-CN');

  function filterItem(element) {
    if (element.matches('button.outline-button')) {
      const matches = !query || element.textContent.toLocaleLowerCase('zh-CN').includes(query);
      element.classList.toggle('search-hidden', !matches);
      return matches;
    }
    if (!element.matches('details')) return false;

    const ownButton = element.querySelector(':scope > summary > .outline-button');
    const ownMatches = !query || ownButton?.textContent.toLocaleLowerCase('zh-CN').includes(query);
    const children = element.querySelector(':scope > .outline-children');
    let descendantMatches = false;
    Array.from(children?.children || []).forEach((child) => {
      if (filterItem(child)) descendantMatches = true;
    });
    const matches = ownMatches || descendantMatches;
    element.classList.toggle('search-hidden', !matches);
    ownButton?.classList.remove('search-hidden');
    if (query && descendantMatches) element.open = true;
    return matches;
  }

  Array.from(outlineEl.children).forEach(filterItem);
}

function updateConfigSummary() {
  const chapterCount = data.courses.reduce((sum, course) => sum + course.chapters.length, 0);
  const sectionCount = data.courses.reduce((sum, course) => sum + course.sections.length, 0);
  const knowledgeCount = data.courses.reduce(
    (sum, course) => sum + course.sections.reduce(
      (sectionSum, section) => sectionSum + (section.knowledge?.length || 0),
      0
    ),
    0
  );
  configSummary.textContent = `${data.courses.length} 门课程 · ${chapterCount} 章 · ${sectionCount} 节 · ${knowledgeCount} 个知识点`;
}

function appendInspectorField(labelText, value, onChange, options = {}) {
  const label = document.createElement('label');
  label.textContent = labelText;
  const input = options.multiline ? document.createElement('textarea') : document.createElement('input');
  if (!options.multiline) input.type = options.type || 'text';
  input.value = value ?? '';
  if (options.step) input.step = options.step;
  if (options.min !== undefined) input.min = options.min;
  if (options.max !== undefined) input.max = options.max;
  input.addEventListener('change', () => onChange(input.value));
  label.appendChild(input);
  inspectorEl.appendChild(label);
  return input;
}

function mutateAndRebuild(mutator, locator, message, focusId = locator?.courseId) {
  const before = snapshotConfig();
  mutator();
  if (!commitHistory(before, message)) {
    renderEditorUI();
    return;
  }
  if (focusId) rebuildSingleCourse(focusId, locator);
  else rebuildAllCourses(locator, null);
}

function renderBoundaryInspector(course, region, type) {
  const title = document.createElement('h2');
  title.textContent = `边界顶点（${region.vertexIds.length}）`;
  inspectorEl.appendChild(title);
  const list = document.createElement('div');
  list.className = 'boundary-list';
  region.vertexIds.forEach((vertexId, index) => {
    const vertex = course.vertices.find((item) => item.id === vertexId);
    const row = document.createElement('div');
    row.className = 'boundary-item';
    const name = document.createElement('span');
    name.textContent = `${index + 1}. ${vertex?.label || vertexId}`;
    name.title = vertexId;
    const up = document.createElement('button');
    up.type = 'button';
    up.textContent = '↑';
    up.title = '前移';
    up.disabled = index === 0;
    up.addEventListener('click', () => {
      mutateAndRebuild(() => {
        [region.vertexIds[index - 1], region.vertexIds[index]] = [
          region.vertexIds[index],
          region.vertexIds[index - 1]
        ];
      }, { type, courseId: course.id, id: region.id }, '边界顶点已前移');
    });
    const down = document.createElement('button');
    down.type = 'button';
    down.textContent = '↓';
    down.title = '后移';
    down.disabled = index === region.vertexIds.length - 1;
    down.addEventListener('click', () => {
      mutateAndRebuild(() => {
        [region.vertexIds[index], region.vertexIds[index + 1]] = [
          region.vertexIds[index + 1],
          region.vertexIds[index]
        ];
      }, { type, courseId: course.id, id: region.id }, '边界顶点已后移');
    });
    const unlink = document.createElement('button');
    unlink.type = 'button';
    unlink.textContent = '×';
    unlink.title = '从此边界移除';
    unlink.className = 'danger';
    unlink.disabled = new Set(region.vertexIds).size <= 3;
    unlink.addEventListener('click', () => {
      if (new Set(region.vertexIds).size <= 3) {
        showToast('边界至少需要 3 个不同顶点');
        return;
      }
      mutateAndRebuild(
        () => region.vertexIds.splice(index, 1),
        { type, courseId: course.id, id: region.id },
        '顶点已从边界移除'
      );
    });
    row.append(name, up, down, unlink);
    list.appendChild(row);
  });
  inspectorEl.appendChild(list);
}

function addAction(label, handler, options = {}) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  if (options.title) button.title = options.title;
  if (options.danger) button.classList.add('danger');
  button.addEventListener('click', handler);
  selectionActionsEl.appendChild(button);
}

function renderSelectionActions() {
  selectionActionsEl.innerHTML = '';
  const item = selectedNode;
  if (!item?.course) return;
  if (item.type === 'course') {
    addAction('＋新建章', () => startBoundaryDraft('chapter', item.course));
    addAction('删除课程', () => deleteSelection(), { danger: true });
  } else if (item.type === 'chapter') {
    addAction('＋新建小节', () => startBoundaryDraft('section', item.course, item.chapter));
    addAction(
      '＋单区域边界点',
      () => startBoundaryPointPlacement('chapter', item.course, item.chapter, 'single'),
      { title: '只向当前章节边界添加一个点' }
    );
    addAction(
      '＋相邻共同点',
      () => startBoundaryPointPlacement('chapter', item.course, item.chapter, 'shared'),
      { title: '在当前章节与相邻章节的共同边界上添加共享点' }
    );
    addAction('删除章', () => deleteSelection(), { danger: true });
  } else if (item.type === 'section') {
    addAction('＋知识点', () => startKnowledgePlacement(item.course, item.section));
    addAction(
      '＋单区域边界点',
      () => startBoundaryPointPlacement('section', item.course, item.section, 'single'),
      { title: '只向当前小节边界添加一个点' }
    );
    addAction(
      '＋相邻共同点',
      () => startBoundaryPointPlacement('section', item.course, item.section, 'shared'),
      { title: '在当前小节与相邻小节的共同边界上添加共享点' }
    );
    addAction('删除小节', () => deleteSelection(), { danger: true });
  } else if (item.type === 'knowledge') {
    addAction('删除知识点', () => deleteSelection(), { danger: true });
  } else if (item.type === 'control' || item.type === 'point') {
    addAction('删除顶点', () => deleteSelection(), { danger: true });
  }
}

function renderEditorUI() {
  renderOutline();
  updateConfigSummary();
  updateHistoryUI();
  renderSelectionActions();
  inspectorEl.innerHTML = '';
  const item = selectedNode;
  if (!item?.course) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = '从左侧结构树或球面选择一个对象。';
    inspectorEl.appendChild(empty);
    return;
  }

  const heading = document.createElement('h2');
  const headings = {
    course: '课程属性',
    chapter: '章节属性',
    section: '小节属性',
    knowledge: '知识点属性',
    control: '球面顶点属性',
    point: '球面顶点属性'
  };
  heading.textContent = headings[item.type] || '属性';
  inspectorEl.appendChild(heading);

  if (item.type === 'course') {
    appendInspectorField('课程名称', item.course.title, (value) => {
      mutateAndRebuild(
        () => { item.course.title = value.trim() || '未命名课程'; },
        { type: 'course', courseId: item.course.id },
        '课程名称已更新'
      );
    });
    appendInspectorField('课程颜色', item.course.color, (value) => {
      mutateAndRebuild(
        () => { item.course.color = value; },
        { type: 'course', courseId: item.course.id },
        '课程颜色已更新'
      );
    }, { type: 'color' });
    appendInspectorField('说明', item.course.description || '', (value) => {
      mutateAndRebuild(
        () => { item.course.description = value; },
        { type: 'course', courseId: item.course.id },
        '课程说明已更新'
      );
    }, { multiline: true });
  }

  if (item.type === 'chapter') {
    appendInspectorField('章节名称', item.chapter.title, (value) => {
      mutateAndRebuild(
        () => { item.chapter.title = value.trim() || '未命名章节'; },
        { type: 'chapter', courseId: item.course.id, id: item.chapter.id },
        '章节名称已更新'
      );
    });
    appendInspectorField('说明', item.chapter.description || '', (value) => {
      mutateAndRebuild(
        () => { item.chapter.description = value; },
        { type: 'chapter', courseId: item.course.id, id: item.chapter.id },
        '章节说明已更新'
      );
    }, { multiline: true });
    renderBoundaryInspector(item.course, item.chapter, 'chapter');
  }

  if (item.type === 'section') {
    appendInspectorField('小节名称', item.section.title, (value) => {
      mutateAndRebuild(
        () => { item.section.title = value.trim() || '未命名小节'; },
        { type: 'section', courseId: item.course.id, id: item.section.id },
        '小节名称已更新'
      );
    });
    appendInspectorField('说明', item.section.description || '', (value) => {
      mutateAndRebuild(
        () => { item.section.description = value; },
        { type: 'section', courseId: item.course.id, id: item.section.id },
        '小节说明已更新'
      );
    }, { multiline: true });
    renderBoundaryInspector(item.course, item.section, 'section');
  }

  if (item.point) {
    appendInspectorField(
      item.type === 'knowledge' ? '知识点名称' : '顶点名称',
      item.point.label,
      (value) => {
        const locator = selectionLocator(item);
        mutateAndRebuild(
          () => {
            if (item.type === 'knowledge') materializeGeneratedKnowledge(item.section);
            item.point.label = value.trim() || item.point.id;
          },
          locator,
          '名称已更新'
        );
      }
    );
    const coordinateGrid = document.createElement('div');
    coordinateGrid.className = 'coordinate-grid';
    inspectorEl.appendChild(coordinateGrid);
    const phi = appendInspectorField('phi（弧度）', item.point.phi, () => {}, {
      type: 'number', step: '0.01', min: -Math.PI, max: Math.PI
    });
    const psi = appendInspectorField('psi（弧度）', item.point.psi, () => {}, {
      type: 'number', step: '0.01', min: -Math.PI / 2, max: Math.PI / 2
    });
    coordinateGrid.append(phi.parentElement, psi.parentElement);
    const commitCoordinates = () => {
      const nextPhi = Number(phi.value);
      const nextPsi = Number(psi.value);
      if (!Number.isFinite(nextPhi) || !Number.isFinite(nextPsi)) {
        showToast('请输入有效坐标');
        return;
      }
      const locator = selectionLocator(item);
      const courseGroup = courseGroups.find((entry) => entry.course === item.course);
      if (!courseGroup) return;
      const candidate = pointFromAngles(
        item.point.id,
        item.point.label,
        nextPhi,
        nextPsi,
        radius
      );
      let wasLimited = false;
      mutateAndRebuild(() => {
        const result = applyPointMove(
          courseGroup,
          item.point,
          sourceToVector(candidate, radius)
        );
        wasLimited = result.limited;
      }, locator, '坐标已更新');
      if (wasLimited) showToast('知识点不能移出所属小节，已限制在小节边界内');
    };
    phi.onchange = commitCoordinates;
    psi.onchange = commitCoordinates;
    if (item.type === 'knowledge') {
      appendInspectorField('说明', item.point.description || '', (value) => {
        mutateAndRebuild(
          () => {
            materializeGeneratedKnowledge(item.section);
            item.point.description = value;
          },
          selectionLocator(item),
          '知识点说明已更新'
        );
      }, { multiline: true });
    }
  }
}

function sectionNeighborText(course, section) {
  const siblings = course.sections.filter((item) => item.chapterId === section.chapterId);
  const index = siblings.findIndex((item) => item.id === section.id);
  const previous = index > 0 ? siblings[index - 1].title : '本章起点';
  const next = index >= 0 && index < siblings.length - 1 ? siblings[index + 1].title : '本章终点';
  return `前置：${previous}\n后接：${next}`;
}

function describeSelection(item) {
  if (!item?.type) return '未选择节点';
  if (item.type === 'course') return `课程：${item.course.title}`;
  if (item.type === 'chapter') return `章节：${item.chapter.title}\n小节数：${item.chapter.sectionIds.length}`;
  if (item.type === 'section') return `小节：${item.section.title}\n${sectionNeighborText(item.course, item.section)}`;
  if (item.type === 'knowledge') return `知识点：${item.section.title} / ${item.point.label}\n${sectionNeighborText(item.course, item.section)}`;
  if (item.type === 'control') {
    const count = item.course.__vertexUsage.get(item.point.id)?.size || 0;
    return `可调交点：${item.point.label}\n关联区域：${count} 个`;
  }
  if (item.type === 'point') return `外层点：${item.point.label}`;
  return '未选择节点';
}

function selectNode(item, object = null) {
  if (!item?.type) return;
  selectedNode = { ...item, object };
  selectionText.textContent = describeSelection(item);
  if (item.point) {
    phiInput.value = item.point.phi || 0;
    psiInput.value = item.point.psi || 0;
  }
  editorEl.classList.toggle('hidden', !editMode || !item.point);
  renderEditorUI();
}

function setPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
}

function raycastScene(event) {
  setPointer(event);
  const visible = courseGroups.filter((item) => item.group.visible);
  const targets = visible.flatMap((item) => [
    ...(editMode ? item.handles.children : []),
    ...item.knowledge.children,
    ...(editMode ? [] : item.outer.children),
    item.base
  ]);
  const hit = raycaster.intersectObjects(targets, false)[0];
  if (!hit) return null;
  const courseGroup = visible.find((item) => item.course === hit.object.userData?.course);
  if (!courseGroup) return null;
  let item = hit.object.userData;
  if (hit.object === courseGroup.base && selectedCourse === courseGroup) {
    const localDirection = courseGroup.group.worldToLocal(hit.point.clone()).normalize();
    if (currentLodLevel >= 2) {
      const section = findRegionAtDirection(courseGroup.course, courseGroup.course.sections, localDirection);
      if (section) item = { type: 'section', course: courseGroup.course, section };
    } else if (currentLodLevel === 1) {
      const chapter = findRegionAtDirection(courseGroup.course, courseGroup.course.chapters, localDirection);
      if (chapter) item = { type: 'chapter', course: courseGroup.course, chapter };
    }
  }
  return { hit, courseGroup, item };
}

function syncPointObjects(courseGroup, point) {
  courseGroup.outer.children.forEach((child) => {
    if (child.userData.point === point) child.position.copy(sourceToVector(point, radius + 0.82));
  });
  courseGroup.handles.children.forEach((child) => {
    if (child.userData.point === point) child.position.copy(sourceToVector(point, radius + 0.46));
  });
  courseGroup.knowledge.children.forEach((child) => {
    if (child.userData.point === point) child.position.copy(sourceToVector(point, radius + 0.34));
  });
  courseGroup.knowledgeLabels.children.forEach((child) => {
    if (child.userData.point === point) {
      child.position.copy(sourceToVector(point, radius + 0.48));
      child.userData.anchorLocal = child.position.clone();
    }
  });
}

function applyPointMove(courseGroup, point, localVector, queueRebuild = false) {
  const isSurfaceVertex = courseGroup.course.__vertexMap.has(point.id);
  let nextVector = localVector.clone().normalize().multiplyScalar(radius);
  let limited = false;
  if (!isSurfaceVertex) {
    const section = courseGroup.course.sections.find((item) => item.knowledge?.includes(point));
    if (section) {
      const constrained = constrainKnowledgeDirection(
        courseGroup.course,
        section,
        sourceToVector(point, 1),
        nextVector
      );
      nextVector = constrained.direction.multiplyScalar(radius);
      limited = constrained.limited;
    }
  }
  const changed = sourceToVector(point, 1).angleTo(nextVector) > 1e-7;
  if (!changed) return { surfaceChanged: false, changed: false, limited };
  if (!isSurfaceVertex) {
    const section = courseGroup.course.sections.find((item) => item.knowledge?.includes(point));
    materializeGeneratedKnowledge(section);
    point.manual = true;
  }
  vectorToSource(nextVector, point, radius);
  syncPointObjects(courseGroup, point);
  if (isSurfaceVertex) {
    invalidateCourseLayouts(courseGroup.course);
    updatePreview(courseGroup);
    if (queueRebuild) queueCourseRebuild(courseGroup);
  }
  return { surfaceChanged: isSurfaceVertex, changed: true, limited };
}

function setEditMode(next) {
  editMode = Boolean(next);
  editBtn.classList.toggle('active', editMode);
  editBtn.setAttribute('aria-pressed', editMode ? 'true' : 'false');
  editorEl.classList.toggle('hidden', !editMode || !selectedNode?.point);
}

function nextEntityId(course, type) {
  const base = createEntityId(course, type);
  const extraIds = new Set(placementMode?.newVertices?.map((point) => point.id) || []);
  if (!extraIds.has(base)) return base;
  let index = 2;
  let candidate = `${base}-${index}`;
  while (extraIds.has(candidate)) {
    index += 1;
    candidate = `${base}-${index}`;
  }
  return candidate;
}

function pointFromLocalVector(course, localVector, type = 'vertex') {
  const normalized = localVector.clone().normalize();
  const phi = Math.atan2(normalized.z, normalized.x);
  const psi = Math.asin(THREE.MathUtils.clamp(normalized.y, -1, 1));
  const id = nextEntityId(course, type);
  const label = type === 'knowledge'
    ? `知识点 ${course.sections.reduce((sum, section) => sum + section.knowledge.length, 0) + 1}`
    : `P${course.vertices.length + (placementMode?.newVertices?.length || 0) + 1}`;
  return pointFromAngles(id, label, phi, psi, radius);
}

function renderPlacementUI() {
  placementBar.classList.toggle('hidden', !placementMode);
  if (!placementMode) return;
  const isDraft = placementMode.type === 'boundary-draft';
  draftNameField.classList.toggle('hidden', !isDraft);
  finishPlacementBtn.classList.toggle('hidden', !isDraft);
  if (isDraft) {
    const noun = placementMode.regionType === 'chapter' ? '章节' : '小节';
    placementTitle.textContent = `新建${noun}边界`;
    draftNameInput.value = placementMode.title;
    placementHint.textContent = `依次点击现有顶点或球面空白处，当前 ${placementMode.vertexIds.length} 个点`;
  } else if (placementMode.type === 'boundary-point') {
    const isShared = placementMode.boundaryScope === 'shared';
    placementTitle.textContent = isShared ? '添加相邻区域共同边界点' : '添加单区域边界点';
    placementHint.textContent = isShared
      ? '点击当前区域与相邻区域的共同边界线，新点会吸附到边界并同时加入两个区域'
      : '点击球面，新点只会插入当前区域最近的边界线段';
  } else if (placementMode.type === 'knowledge') {
    placementTitle.textContent = '新建知识点';
    placementHint.textContent = '请在目标小节区域内点击球面';
  } else {
    placementTitle.textContent = '新建球面顶点';
    placementHint.textContent = '点击球面放置一个可复用顶点';
  }
}

function renderDraft() {
  courseGroups.forEach((courseGroup) => clearDynamicGroup(courseGroup.draft));
  if (placementMode?.type !== 'boundary-draft') return;
  const courseGroup = courseGroups.find((item) => item.course.id === placementMode.courseId);
  if (!courseGroup) return;
  const pointsById = new Map(courseGroup.course.vertices.map((point) => [point.id, point]));
  placementMode.newVertices.forEach((point) => pointsById.set(point.id, point));
  const vectors = placementMode.vertexIds
    .map((id) => pointsById.get(id))
    .filter(Boolean)
    .map((point) => sourceToVector(point, radius + 0.56));

  vectors.forEach((position) => {
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xff8a00,
      emissiveIntensity: 0.75,
      roughness: 0.35,
      depthTest: false
    });
    const dot = new THREE.Mesh(handleGeometry, material);
    dot.position.copy(position);
    dot.userData.sharedGeometry = true;
    dot.renderOrder = 25;
    courseGroup.draft.add(dot);
  });

  if (vectors.length >= 2) {
    const positions = [];
    const edgeCount = vectors.length >= 3 ? vectors.length : vectors.length - 1;
    for (let index = 0; index < edgeCount; index += 1) {
      const start = vectors[index];
      const end = vectors[(index + 1) % vectors.length];
      for (let step = 0; step <= 16; step += 1) {
        const point = slerpUnit(start, end, step / 16).multiplyScalar(radius + 0.55);
        positions.push(point.x, point.y, point.z);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({
      color: 0xff7a00,
      depthTest: false,
      transparent: true,
      opacity: 0.95
    });
    const line = new THREE.Line(geometry, material);
    line.renderOrder = 24;
    courseGroup.draft.add(line);
  }
}

function cancelPlacement(notify = true) {
  placementMode = null;
  renderPlacementUI();
  renderDraft();
  if (notify) showToast('已取消放置');
}

function startBoundaryDraft(regionType, course, chapter = null) {
  cancelPlacement(false);
  focusCourse(course.id);
  setEditMode(true);
  placementMode = {
    type: 'boundary-draft',
    regionType,
    courseId: course.id,
    chapterId: chapter?.id || null,
    title: regionType === 'chapter'
      ? `新章节 ${course.chapters.length + 1}`
      : `新小节 ${course.sections.filter((section) => section.chapterId === chapter?.id).length + 1}`,
    vertexIds: [],
    newVertices: []
  };
  renderPlacementUI();
  renderDraft();
}

function startBoundaryPointPlacement(regionType, course, region, boundaryScope = 'single') {
  const regions = regionType === 'chapter' ? course.chapters : course.sections;
  if (
    boundaryScope === 'shared'
    && sharedBoundaryNeighbors(region, regions).length === 0
  ) {
    showToast('当前区域没有可添加共同点的相邻区域');
    return;
  }
  cancelPlacement(false);
  focusCourse(course.id);
  setEditMode(true);
  placementMode = {
    type: 'boundary-point',
    regionType,
    courseId: course.id,
    regionId: region.id,
    boundaryScope
  };
  renderPlacementUI();
}

function startKnowledgePlacement(course, section) {
  cancelPlacement(false);
  focusCourse(course.id);
  setEditMode(true);
  placementMode = {
    type: 'knowledge',
    courseId: course.id,
    sectionId: section.id
  };
  renderPlacementUI();
}

function startFreeVertexPlacement(course) {
  cancelPlacement(false);
  focusCourse(course.id);
  setEditMode(true);
  placementMode = { type: 'vertex', courseId: course.id };
  renderPlacementUI();
}

function finishBoundaryDraft() {
  if (placementMode?.type !== 'boundary-draft') return;
  placementMode.title = draftNameInput.value.trim() || placementMode.title;
  const uniqueIds = [...new Set(placementMode.vertexIds)];
  if (uniqueIds.length < 3) {
    showToast('闭合边界至少需要 3 个不同顶点');
    return;
  }
  const course = data.courses.find((item) => item.id === placementMode.courseId);
  const chapter = course?.chapters.find((item) => item.id === placementMode.chapterId);
  if (!course || (placementMode.regionType === 'section' && !chapter)) {
    showToast('目标课程或章节已不存在');
    cancelPlacement(false);
    return;
  }
  const pointsById = new Map(course.vertices.map((point) => [point.id, point]));
  placementMode.newVertices.forEach((point) => pointsById.set(point.id, point));
  const boundaryDirections = uniqueIds
    .map((id) => pointsById.get(id))
    .filter(Boolean)
    .map((point) => sourceToVector(point, 1));
  const areaNormal = new THREE.Vector3();
  boundaryDirections.forEach((point, index) => {
    areaNormal.add(new THREE.Vector3().crossVectors(
      point,
      boundaryDirections[(index + 1) % boundaryDirections.length]
    ));
  });
  if (boundaryDirections.length < 3 || areaNormal.lengthSq() < 1e-5) {
    showToast('这些顶点无法构成有效球面区域，请调整点位或顺序');
    return;
  }
  if (placementMode.regionType === 'section') {
    const chapterBoundary = sampleClosedBoundary(course, chapter, 8, 1);
    const outside = uniqueIds.some((id) => {
      const point = pointsById.get(id);
      return !point || !sphericalContains(sourceToVector(point, 1), chapterBoundary);
    });
    if (outside) {
      showToast('小节边界点必须位于所属章节内');
      return;
    }
  }
  const mode = placementMode;
  const before = snapshotConfig();
  course.vertices.push(...mode.newVertices);
  let locator;
  if (mode.regionType === 'chapter') {
    const id = createEntityId(course, 'chapter');
    course.chapters.push({
      id,
      title: mode.title,
      vertexIds: uniqueIds,
      sectionIds: []
    });
    locator = { type: 'chapter', courseId: course.id, id };
  } else {
    const id = createEntityId(course, 'section');
    course.sections.push({
      id,
      title: mode.title,
      chapterId: chapter.id,
      vertexIds: uniqueIds,
      knowledge: []
    });
    chapter.sectionIds.push(id);
    locator = { type: 'section', courseId: course.id, id };
  }
  cancelPlacement(false);
  commitHistory(before, mode.regionType === 'chapter' ? '章节已创建' : '小节已创建');
  rebuildSingleCourse(course.id, locator);
}

function closestBoundaryInsertionIndex(course, region, direction) {
  let bestIndex = region.vertexIds.length;
  let bestDot = -Infinity;
  region.vertexIds.forEach((vertexId, index) => {
    const start = course.__vertexMap.get(vertexId);
    const end = course.__vertexMap.get(region.vertexIds[(index + 1) % region.vertexIds.length]);
    if (!start || !end) return;
    const a = sourceToVector(start, 1);
    const b = sourceToVector(end, 1);
    for (let step = 0; step <= 12; step += 1) {
      const dot = slerpUnit(a, b, step / 12).dot(direction);
      if (dot > bestDot) {
        bestDot = dot;
        bestIndex = index + 1;
      }
    }
  });
  return bestIndex;
}

function addDraftVertex(course, point, isNew = false) {
  if (placementMode.vertexIds.includes(point.id)) {
    showToast('这个顶点已经在当前边界中');
    return;
  }
  placementMode.vertexIds.push(point.id);
  if (isNew) placementMode.newVertices.push(point);
  renderPlacementUI();
  renderDraft();
}

function handlePlacement(result) {
  if (!placementMode) return false;
  const { hit, courseGroup, item } = result;
  if (courseGroup.course.id !== placementMode.courseId) {
    showToast('请在当前目标课程球上操作');
    return true;
  }

  if (placementMode.type === 'boundary-draft' && item.type === 'control') {
    addDraftVertex(courseGroup.course, item.point);
    return true;
  }
  if (hit.object !== courseGroup.base) {
    showToast('请点击球面空白区域');
    return true;
  }
  const direction = courseGroup.group.worldToLocal(hit.point.clone()).normalize();

  if (placementMode.type === 'boundary-draft') {
    addDraftVertex(
      courseGroup.course,
      pointFromLocalVector(courseGroup.course, direction, 'vertex'),
      true
    );
    return true;
  }

  if (placementMode.type === 'boundary-point') {
    const course = courseGroup.course;
    const regions = placementMode.regionType === 'chapter' ? course.chapters : course.sections;
    const region = regions.find((entry) => entry.id === placementMode.regionId);
    if (!region) {
      cancelPlacement(false);
      return true;
    }
    const isShared = placementMode.boundaryScope === 'shared';
    const sharedMatch = isShared
      ? findClosestSharedBoundary(course, region, regions, direction)
      : null;
    if (isShared && (!sharedMatch || sharedMatch.distance > 0.16)) {
      showToast('请点击当前区域与相邻区域的共同边界线');
      return true;
    }
    const pointDirection = sharedMatch?.direction || direction;
    const affectedRegions = sharedMatch ? [region, sharedMatch.neighbor] : [region];
    if (placementMode.regionType === 'section') {
      const outsideParent = affectedRegions.some((affectedRegion) => {
        const chapter = course.chapters.find((entry) => entry.id === affectedRegion.chapterId);
        const chapterBoundary = chapter ? sampleClosedBoundary(course, chapter, 8, 1) : [];
        return !chapter || !sphericalContains(pointDirection, chapterBoundary);
      });
      if (outsideParent) {
        showToast(isShared
          ? '共同边界点必须同时位于两个小节所属章节内'
          : '小节边界点必须位于所属章节内');
        return true;
      }
    }
    const before = snapshotConfig();
    const point = pointFromLocalVector(course, pointDirection, 'vertex');
    course.vertices.push(point);
    if (sharedMatch) {
      region.vertexIds.splice(sharedMatch.regionInsertionIndex, 0, point.id);
      sharedMatch.neighbor.vertexIds.splice(sharedMatch.neighborInsertionIndex, 0, point.id);
    } else {
      const insertionIndex = closestBoundaryInsertionIndex(course, region, pointDirection);
      region.vertexIds.splice(insertionIndex, 0, point.id);
    }
    const locator = {
      type: placementMode.regionType,
      courseId: course.id,
      id: region.id
    };
    cancelPlacement(false);
    commitHistory(before, sharedMatch
      ? `已添加与“${sharedMatch.neighbor.title}”共用的边界点`
      : '单区域边界顶点已添加');
    rebuildSingleCourse(course.id, locator);
    return true;
  }

  if (placementMode.type === 'knowledge') {
    const course = courseGroup.course;
    const section = course.sections.find((entry) => entry.id === placementMode.sectionId);
    const clickedSection = findRegionAtDirection(course, course.sections, direction);
    if (!section || clickedSection?.id !== section.id) {
      showToast('知识点必须放在目标小节区域内');
      return true;
    }
    const before = snapshotConfig();
    const point = pointFromLocalVector(course, direction, 'knowledge');
    section.knowledge.push(point);
    const locator = {
      type: 'knowledge',
      courseId: course.id,
      sectionId: section.id,
      id: point.id
    };
    cancelPlacement(false);
    commitHistory(before, '知识点已创建');
    rebuildSingleCourse(course.id, locator);
    return true;
  }

  if (placementMode.type === 'vertex') {
    const course = courseGroup.course;
    const before = snapshotConfig();
    const point = pointFromLocalVector(course, direction, 'vertex');
    course.vertices.push(point);
    const locator = { type: 'vertex', courseId: course.id, id: point.id };
    cancelPlacement(false);
    commitHistory(before, '球面顶点已创建');
    rebuildSingleCourse(course.id, locator);
    return true;
  }
  return false;
}

function deleteSelection() {
  const item = selectedNode;
  if (!item?.course) return;
  const course = item.course;
  const before = snapshotConfig();
  let locator = { type: 'course', courseId: course.id };
  let message = '';

  if (item.type === 'course') {
    if (!window.confirm(`删除课程“${course.title}”及其全部内容？`)) return;
    data.courses.splice(data.courses.indexOf(course), 1);
    locator = null;
    message = '课程已删除';
  } else if (item.type === 'chapter') {
    const childSections = course.sections.filter((section) => section.chapterId === item.chapter.id);
    if (!window.confirm(`删除“${item.chapter.title}”及其 ${childSections.length} 个小节？`)) return;
    course.chapters.splice(course.chapters.indexOf(item.chapter), 1);
    course.sections = course.sections.filter((section) => section.chapterId !== item.chapter.id);
    message = '章节已删除';
  } else if (item.type === 'section') {
    if (!window.confirm(`删除小节“${item.section.title}”及其知识点？`)) return;
    course.sections.splice(course.sections.indexOf(item.section), 1);
    const chapter = course.chapters.find((entry) => entry.id === item.section.chapterId);
    if (chapter) chapter.sectionIds = chapter.sectionIds.filter((id) => id !== item.section.id);
    locator = chapter ? { type: 'chapter', courseId: course.id, id: chapter.id } : locator;
    message = '小节已删除';
  } else if (item.type === 'knowledge') {
    materializeGeneratedKnowledge(item.section);
    item.section.knowledge.splice(item.section.knowledge.indexOf(item.point), 1);
    locator = { type: 'section', courseId: course.id, id: item.section.id };
    message = '知识点已删除';
  } else if (item.type === 'control' || item.type === 'point') {
    const regions = [...course.chapters, ...course.sections]
      .filter((region) => region.vertexIds.includes(item.point.id));
    const invalidRegion = regions.find((region) => (
      new Set(region.vertexIds.filter((id) => id !== item.point.id)).size < 3
    ));
    if (invalidRegion) {
      showToast(`无法删除：会使“${invalidRegion.title}”少于 3 个边界顶点`);
      return;
    }
    if (regions.length && !window.confirm(`该顶点被 ${regions.length} 个区域引用，确定从所有边界中删除？`)) return;
    regions.forEach((region) => {
      region.vertexIds = region.vertexIds.filter((id) => id !== item.point.id);
    });
    course.vertices.splice(course.vertices.indexOf(item.point), 1);
    message = '顶点已删除';
  } else {
    return;
  }

  commitHistory(before, message);
  if (locator?.courseId) rebuildSingleCourse(locator.courseId, locator);
  else rebuildAllCourses(null, null);
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  const result = raycastScene(event);
  if (!result) return;
  const { hit, courseGroup, item } = result;
  if (handlePlacement(result)) return;
  if (!selectedCourse || selectedCourse.course.id !== courseGroup.course.id) focusCourse(courseGroup.course.id);
  selectNode(item, hit.object);
  if (editMode && item.point) {
    draggingPoint = {
      courseGroup,
      point: item.point,
      startVector: sourceToVector(item.point, radius),
      surfaceChanged: false,
      limited: false,
      before: snapshotConfig()
    };
    controls.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
  }
});

renderer.domElement.addEventListener('pointermove', (event) => {
  if (!draggingPoint) return;
  setPointer(event);
  const center = new THREE.Vector3();
  draggingPoint.courseGroup.group.getWorldPosition(center);
  const hit = new THREE.Vector3();
  const intersects = raycaster.ray.intersectSphere(new THREE.Sphere(center, radius + 0.48), hit);
  if (!intersects) return;
  const localTarget = draggingPoint.courseGroup.group.worldToLocal(hit.clone()).normalize().multiplyScalar(radius);
  const local = event.shiftKey
    ? slerpUnit(draggingPoint.startVector, localTarget, 0.18).multiplyScalar(radius)
    : localTarget;
  const result = applyPointMove(draggingPoint.courseGroup, draggingPoint.point, local);
  draggingPoint.surfaceChanged = result.surfaceChanged || draggingPoint.surfaceChanged;
  draggingPoint.limited = result.limited || draggingPoint.limited;
  phiInput.value = draggingPoint.point.phi;
  psiInput.value = draggingPoint.point.psi;
});

function finishPointerDrag(event) {
  if (!draggingPoint) return;
  if (draggingPoint.surfaceChanged) rebuildCourseVisuals(draggingPoint.courseGroup);
  commitHistory(draggingPoint.before, draggingPoint.surfaceChanged ? '边界顶点已移动' : '知识点已移动');
  if (draggingPoint.limited) showToast('知识点不能移出所属小节，已限制在小节边界内');
  renderEditorUI();
  draggingPoint = null;
  controls.enabled = true;
  if (renderer.domElement.hasPointerCapture(event.pointerId)) renderer.domElement.releasePointerCapture(event.pointerId);
}

renderer.domElement.addEventListener('pointerup', finishPointerDrag);
renderer.domElement.addEventListener('pointercancel', finishPointerDrag);

function updateSelectedPoint() {
  if (!selectedNode?.point || !selectedNode?.course) return null;
  if (!sliderBefore) {
    sliderBefore = snapshotConfig();
    sliderLimited = false;
  }
  const courseGroup = courseGroups.find((item) => item.course === selectedNode.course);
  if (!courseGroup) return null;
  const phi = Number(phiInput.value);
  const psi = Number(psiInput.value);
  const cp = Math.cos(psi);
  const local = new THREE.Vector3(
    radius * cp * Math.cos(phi),
    radius * Math.sin(psi),
    radius * cp * Math.sin(phi)
  );
  const result = applyPointMove(courseGroup, selectedNode.point, local, true);
  sliderLimited = result.limited || sliderLimited;
  if (result.limited) {
    phiInput.value = selectedNode.point.phi;
    psiInput.value = selectedNode.point.psi;
  }
  return result;
}

function importConfigFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.addEventListener('load', () => {
    try {
      const parsed = JSON.parse(String(reader.result));
      const normalized = normalizeConfig(parsed);
      replaceConfig(data, normalized.config);
      undoStack.length = 0;
      redoStack.length = 0;
      savedSnapshot = snapshotConfig();
      cancelPlacement(false);
      rebuildAllCourses(null, null);
      updateHistoryUI();
      const warningText = normalized.warnings.length
        ? `；已自动处理 ${normalized.warnings.length} 个兼容问题`
        : '';
      showToast(`配置导入成功${warningText}`, 4200);
    } catch (error) {
      const details = error instanceof ConfigValidationError
        ? error.messages.slice(0, 6).join('\n')
        : error.message;
      window.alert(`配置导入失败：\n${details}`);
    } finally {
      fileInput.value = '';
    }
  });
  reader.addEventListener('error', () => {
    showToast('无法读取所选文件');
    fileInput.value = '';
  });
  reader.readAsText(file);
}

function exportCurrentConfig() {
  if (placementMode) {
    showToast('请先完成或取消当前放置操作');
    return;
  }
  try {
    const clean = exportConfig(data);
    const content = JSON.stringify(clean, null, 2);
    const blob = new Blob([content], { type: 'application/json;charset=utf-8' });
    const link = document.createElement('a');
    const firstTitle = clean.courses[0]?.title || 'sphere-knowledge-map';
    const safeName = firstTitle.replace(/[\\/:*?"<>|\s]+/g, '-').replace(/^-|-$/g, '');
    link.href = URL.createObjectURL(blob);
    link.download = `${safeName || 'sphere-knowledge-map'}-config.json`;
    link.click();
    URL.revokeObjectURL(link.href);
    savedSnapshot = snapshotConfig();
    updateHistoryUI();
    showToast('配置已导出');
  } catch (error) {
    const details = error instanceof ConfigValidationError
      ? error.messages.slice(0, 6).join('\n')
      : error.message;
    window.alert(`导出失败：\n${details}`);
  }
}

function addCourse() {
  cancelPlacement(false);
  const before = snapshotConfig();
  const course = createEmptyCourse(data);
  data.courses.push(course);
  commitHistory(before, '空白课程已创建');
  rebuildAllCourses({ type: 'course', courseId: course.id }, course.id);
}

function setPanelCollapsed(panel, revealButton, collapsed) {
  panel.classList.toggle('is-collapsed', collapsed);
  revealButton.classList.toggle('visible', collapsed);
  revealButton.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
}

hideLibraryBtn.addEventListener('click', () => setPanelCollapsed(libraryPanel, showLibraryBtn, true));
showLibraryBtn.addEventListener('click', () => setPanelCollapsed(libraryPanel, showLibraryBtn, false));
hideInspectorBtn.addEventListener('click', () => setPanelCollapsed(inspectorPanel, showInspectorBtn, true));
showInspectorBtn.addEventListener('click', () => setPanelCollapsed(inspectorPanel, showInspectorBtn, false));
helpBtn.addEventListener('click', () => helpDialog.showModal());
outlineSearch.addEventListener('input', () => applyOutlineFilter(outlineSearch.value));

overviewBtn.addEventListener('click', () => {
  cancelPlacement(false);
  selectedNode = null;
  focusCourse(null);
  renderEditorUI();
});
editBtn.addEventListener('click', () => setEditMode(!editMode));
addVertexBtn.addEventListener('click', () => {
  if (!selectedCourse) {
    showToast('请先选择一门课程');
    return;
  }
  startFreeVertexPlacement(selectedCourse.course);
});
addCourseBtn.addEventListener('click', addCourse);
importBtn.addEventListener('click', () => {
  const dirty = snapshotConfig() !== savedSnapshot;
  if (dirty && !window.confirm('导入会替换当前未导出的编辑内容，确定继续？')) return;
  fileInput.click();
});
exportBtn.addEventListener('click', exportCurrentConfig);
undoBtn.addEventListener('click', undo);
redoBtn.addEventListener('click', redo);
fileInput.addEventListener('change', () => importConfigFile(fileInput.files?.[0]));
finishPlacementBtn.addEventListener('click', finishBoundaryDraft);
cancelPlacementBtn.addEventListener('click', () => cancelPlacement());
draftNameInput.addEventListener('input', () => {
  if (placementMode?.type === 'boundary-draft') placementMode.title = draftNameInput.value;
});
phiInput.addEventListener('pointerdown', () => {
  sliderBefore = snapshotConfig();
  sliderLimited = false;
});
psiInput.addEventListener('pointerdown', () => {
  sliderBefore = snapshotConfig();
  sliderLimited = false;
});
phiInput.addEventListener('input', updateSelectedPoint);
psiInput.addEventListener('input', updateSelectedPoint);
phiInput.addEventListener('change', () => {
  if (sliderBefore) commitHistory(sliderBefore, '坐标已更新');
  if (sliderLimited) showToast('知识点不能移出所属小节，已限制在小节边界内');
  sliderBefore = null;
  sliderLimited = false;
  renderEditorUI();
});
psiInput.addEventListener('change', () => {
  if (sliderBefore) commitHistory(sliderBefore, '坐标已更新');
  if (sliderLimited) showToast('知识点不能移出所属小节，已限制在小节边界内');
  sliderBefore = null;
  sliderLimited = false;
  renderEditorUI();
});
document.querySelectorAll('[data-nudge]').forEach((button) => {
  button.addEventListener('click', () => {
    if (!selectedNode?.point) return;
    const before = snapshotConfig();
    const step = 0.025;
    if (button.dataset.nudge === 'left') phiInput.value = Number(phiInput.value) - step;
    if (button.dataset.nudge === 'right') phiInput.value = Number(phiInput.value) + step;
    if (button.dataset.nudge === 'up') psiInput.value = Number(psiInput.value) + step;
    if (button.dataset.nudge === 'down') psiInput.value = Number(psiInput.value) - step;
    const result = updateSelectedPoint();
    commitHistory(before, '坐标已微调');
    if (result?.limited) showToast('知识点不能移出所属小节，已限制在小节边界内');
    sliderBefore = null;
    sliderLimited = false;
    renderEditorUI();
  });
});

window.addEventListener('keydown', (event) => {
  const active = document.activeElement;
  const editingText = active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement;
  if (event.key === 'Escape' && placementMode) {
    event.preventDefault();
    cancelPlacement();
    return;
  }
  if (!editingText && !event.ctrlKey && !event.metaKey && event.key.toLowerCase() === 'f') {
    event.preventDefault();
    outlineSearch.focus();
    outlineSearch.select();
    return;
  }
  if (editingText || !(event.ctrlKey || event.metaKey)) return;
  if (event.key.toLowerCase() === 'z') {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
  } else if (event.key.toLowerCase() === 'y') {
    event.preventDefault();
    redo();
  } else if (event.key.toLowerCase() === 's') {
    event.preventDefault();
    exportCurrentConfig();
  }
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (!selectedCourse) {
    courseGroups.forEach((item, index) => {
      item.group.userData.target = overviewPosition(index);
    });
  } else {
    const offset = camera.position.clone().sub(controls.target);
    controls.target.y = isMobileView() ? -3.2 : 0;
    camera.position.copy(controls.target).add(offset);
  }
});

window.addEventListener('beforeunload', (event) => {
  if (snapshotConfig() === savedSnapshot) return;
  event.preventDefault();
  event.returnValue = '';
});

function animate() {
  requestAnimationFrame(animate);
  starField.rotation.y += 0.000035;
  courseGroups.forEach((item) => {
    const target = item.group.userData.target || item.group.position;
    item.group.position.lerp(target, 0.08);
  });
  controls.update();
  updateLod();
  updateSurfaceLabelVisibility();
  renderer.render(scene, camera);
}

renderTabs();
const queryCourse = new URLSearchParams(window.location.search).get('course');
if (queryCourse && data.courses.some((course) => course.id === queryCourse)) focusCourse(queryCourse);
else focusCourse(null);
renderEditorUI();
renderPlacementUI();
if (initialConfig.warnings.length) {
  showToast(`示例配置已自动处理 ${initialConfig.warnings.length} 个兼容问题`, 4200);
}
animate();
