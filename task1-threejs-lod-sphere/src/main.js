import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import data from './data/course-data.json';
import {
  createSemanticFields,
  createSemanticMaterial,
  disposeSemanticFields,
  findRegionAtDirection,
  invalidateCourseLayouts,
  prepareCourse,
  readableTextColor,
  regionLayout,
  sampleClosedBoundary,
  slerpUnit,
  smoothstep,
  sourceToVector,
  updateMaterialFields,
  vectorToSource
} from './spherical-field.js';
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
const overviewDistance = 122;
const overviewSpacing = 26.5;
const focusDistance = 55;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef2f7);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1200);
camera.position.set(0, 10, overviewDistance);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
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

const outerGeometry = new THREE.SphereGeometry(0.17, 18, 14);
const handleGeometry = new THREE.SphereGeometry(0.22, 18, 14);
const knowledgeGeometry = new THREE.SphereGeometry(0.12, 16, 12);
const knowledgeLabelCache = new Map();

function isMobileView() {
  return window.innerWidth < 720;
}

function currentOverviewDistance() {
  return isMobileView() ? 150 : overviewDistance;
}

function overviewPosition(index) {
  if (!isMobileView()) return new THREE.Vector3((index - 1.5) * overviewSpacing, 0, 0);
  const column = index % 2 === 0 ? -1 : 1;
  const row = index < 2 ? 18 : -5;
  return new THREE.Vector3(column * 10.5, row, 0);
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
  let width = Math.min(options.maxWidth || 6.2, available);
  width = Math.max(options.minWidth || 0.9, width);
  let height = width / aspect;
  const maxHeight = Math.max(0.45, available * 0.62);
  if (height > maxHeight) {
    height = maxHeight;
    width = height * aspect;
  }
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    alphaTest: 0.025,
    depthTest: true,
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
    const { texture, aspect } = makeTextTexture(text, '#0f172a', 3);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: true, depthWrite: false });
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
}

function buildOuterPoints(courseGroup) {
  const { course, outer } = courseGroup;
  const stride = Math.max(1, Math.ceil(course.vertices.length / 8));
  course.vertices.forEach((point, index) => {
    if (index % stride !== 0 && index !== course.vertices.length - 1) return;
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
  const layout = regionLayout(course, section);
  const boundary = sampleClosedBoundary(course, section, 6, 1);
  if (!section.knowledge.length) {
    section.knowledge = [0, 1, 2].map((index) => ({
      id: `${section.id}-knowledge-${index + 1}`,
      label: `K${index + 1}`,
      generated: true
    }));
  }
  section.knowledge.forEach((point, index) => {
    if (point.manual) return;
    const target = boundary[Math.floor(((index + 0.65) / section.knowledge.length) * boundary.length) % boundary.length];
    const direction = target ? slerpUnit(layout.anchor, target, index === 0 ? 0.18 : 0.34) : layout.anchor;
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
    const label = makeSurfaceLabel(chapter.title, readableTextColor(chapter.__color), layout, {
      maxChars: 8,
      minWidth: 3.0,
      maxWidth: 8.6,
      radialOffset: 0.25,
      priority: 3,
      renderOrder: 7
    });
    label.userData = { ...label.userData, type: 'chapter', course, chapter };
    chapterLabels.add(label);
  });

  course.sections.forEach((section) => {
    const layout = regionLayout(course, section);
    const label = makeSurfaceLabel(section.title, readableTextColor(section.__color), layout, {
      maxChars: 9,
      minWidth: 1.3,
      maxWidth: 5.3,
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
  course.edges.forEach((edge) => {
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
  prepareCourse(course);
  buildVertexUsage(course);

  const group = new THREE.Group();
  group.position.copy(overviewPosition(index));
  group.userData.target = group.position.clone();
  root.add(group);

  const fields = createSemanticFields(course, renderer);
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
  group.add(outer, chapterLabels, sectionLabels, knowledge, knowledgeLabels, handles);

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

  const title = makeBillboardLabel(course.title, '#101820', 1.55, false);
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

function rebuildCourseVisuals(courseGroup) {
  invalidateCourseLayouts(courseGroup.course);
  const nextFields = createSemanticFields(courseGroup.course, renderer);
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
}

function updateLayerBlend(distance) {
  const regionReveal = 1 - smoothstep(78, 92, distance);
  const sectionBlend = 1 - smoothstep(43, 56, distance);
  const knowledgeAlpha = 1 - smoothstep(24, 31, distance);
  const outerAlpha = smoothstep(24, 31, distance);
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

function labelScreenRect(label, courseGroup) {
  const material = Array.isArray(label.material) ? label.material[0] : label.material;
  if (!label.userData?.isSurfaceLabel || !material || material.opacity < 0.12) return null;
  const localAnchor = label.userData.anchorLocal;
  if (!localAnchor) return null;
  const world = courseGroup.group.localToWorld(localAnchor.clone());
  const center = new THREE.Vector3();
  courseGroup.group.getWorldPosition(center);
  const normal = world.clone().sub(center).normalize();
  const view = camera.position.clone().sub(world).normalize();
  if (normal.dot(view) < 0.25) return null;
  const projected = world.clone().project(camera);
  if (Math.abs(projected.x) > 1.08 || Math.abs(projected.y) > 1.08 || projected.z < -1 || projected.z > 1) return null;
  const size = label.userData.labelSize || { width: 1, height: 0.4 };
  const distance = Math.max(1, camera.position.distanceTo(world));
  const pixelsPerWorld = renderer.domElement.clientHeight
    / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * distance);
  return {
    x: (projected.x * 0.5 + 0.5) * renderer.domElement.clientWidth,
    y: (-projected.y * 0.5 + 0.5) * renderer.domElement.clientHeight,
    w: size.width * pixelsPerWorld * 1.08,
    h: size.height * pixelsPerWorld * 1.25,
    priority: label.userData.labelPriority || 0,
    opacity: material.opacity,
    label
  };
}

function rectsOverlap(a, b) {
  return Math.abs(a.x - b.x) * 2 < a.w + b.w && Math.abs(a.y - b.y) * 2 < a.h + b.h;
}

function updateSurfaceLabelVisibility() {
  const entries = [];
  courseGroups.forEach((courseGroup) => {
    if (!courseGroup.group.visible) return;
    courseGroup.group.updateMatrixWorld();
    [
      ...courseGroup.chapterLabels.children,
      ...courseGroup.sectionLabels.children,
      ...courseGroup.knowledgeLabels.children
    ].forEach((label) => {
      const rect = labelScreenRect(label, courseGroup);
      if (!rect) {
        label.visible = false;
        return;
      }
      entries.push(rect);
    });
  });
  entries.sort((a, b) => b.priority - a.priority || b.opacity - a.opacity || b.w * b.h - a.w * a.h);
  const accepted = [];
  entries.forEach((entry) => {
    const overlaps = accepted.some((item) => rectsOverlap(entry, item));
    entry.label.visible = !overlaps;
    if (!overlaps) accepted.push(entry);
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
    courseTitle.textContent = '四个教材';
    selectionText.textContent = '未选择节点';
    if (changed) {
      controls.target.set(0, 0, 0);
      camera.position.set(0, 10, currentOverviewDistance());
    }
  }
  renderTabs();
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
  if (item.type === 'course') return `教材：${item.course.title}`;
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
    ...item.outer.children,
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
  vectorToSource(localVector, point, radius);
  const isSurfaceVertex = courseGroup.course.__vertexMap.has(point.id);
  if (!isSurfaceVertex) point.manual = true;
  syncPointObjects(courseGroup, point);
  if (isSurfaceVertex) {
    invalidateCourseLayouts(courseGroup.course);
    updatePreview(courseGroup);
    if (queueRebuild) queueCourseRebuild(courseGroup);
  }
  return isSurfaceVertex;
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  const result = raycastScene(event);
  if (!result) return;
  const { hit, courseGroup, item } = result;
  if (!selectedCourse || selectedCourse.course.id !== courseGroup.course.id) focusCourse(courseGroup.course.id);
  selectNode(item, hit.object);
  if (editMode && item.point) {
    draggingPoint = {
      courseGroup,
      point: item.point,
      startVector: sourceToVector(item.point, radius),
      surfaceChanged: false
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
  draggingPoint.surfaceChanged = applyPointMove(draggingPoint.courseGroup, draggingPoint.point, local) || draggingPoint.surfaceChanged;
  phiInput.value = draggingPoint.point.phi;
  psiInput.value = draggingPoint.point.psi;
});

function finishPointerDrag(event) {
  if (!draggingPoint) return;
  if (draggingPoint.surfaceChanged) rebuildCourseVisuals(draggingPoint.courseGroup);
  draggingPoint = null;
  controls.enabled = true;
  if (renderer.domElement.hasPointerCapture(event.pointerId)) renderer.domElement.releasePointerCapture(event.pointerId);
}

renderer.domElement.addEventListener('pointerup', finishPointerDrag);
renderer.domElement.addEventListener('pointercancel', finishPointerDrag);

function updateSelectedPoint() {
  if (!selectedNode?.point || !selectedNode?.course) return;
  const courseGroup = courseGroups.find((item) => item.course === selectedNode.course);
  if (!courseGroup) return;
  const phi = Number(phiInput.value);
  const psi = Number(psiInput.value);
  const cp = Math.cos(psi);
  const local = new THREE.Vector3(
    radius * cp * Math.cos(phi),
    radius * Math.sin(psi),
    radius * cp * Math.sin(phi)
  );
  applyPointMove(courseGroup, selectedNode.point, local, true);
}

overviewBtn.addEventListener('click', () => focusCourse(null));
editBtn.addEventListener('click', () => {
  editMode = !editMode;
  editBtn.classList.toggle('active', editMode);
  editBtn.setAttribute('aria-pressed', editMode ? 'true' : 'false');
  editorEl.classList.toggle('hidden', !editMode);
});
phiInput.addEventListener('input', updateSelectedPoint);
psiInput.addEventListener('input', updateSelectedPoint);
document.querySelectorAll('[data-nudge]').forEach((button) => {
  button.addEventListener('click', () => {
    if (!selectedNode?.point) return;
    const step = 0.025;
    if (button.dataset.nudge === 'left') phiInput.value = Number(phiInput.value) - step;
    if (button.dataset.nudge === 'right') phiInput.value = Number(phiInput.value) + step;
    if (button.dataset.nudge === 'up') psiInput.value = Number(psiInput.value) + step;
    if (button.dataset.nudge === 'down') psiInput.value = Number(psiInput.value) - step;
    updateSelectedPoint();
  });
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

function animate() {
  requestAnimationFrame(animate);
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
animate();
