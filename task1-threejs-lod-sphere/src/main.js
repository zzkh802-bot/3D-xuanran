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
sceneEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 14;
controls.maxDistance = 170;

scene.add(new THREE.HemisphereLight(0xffffff, 0x768394, 2.35));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(30, 35, 28);
scene.add(keyLight);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const root = new THREE.Group();
scene.add(root);

const courseGroups = [];
let selectedCourse = null;
let selectedNode = null;
let editMode = false;
let draggingPoint = null;

function colorToThree(color, offset = 0) {
  const c = new THREE.Color(color);
  if (offset > 0) c.lerp(new THREE.Color(0xffffff), offset);
  if (offset < 0) c.lerp(new THREE.Color(0x101820), Math.abs(offset));
  return c;
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
  ctx.fillStyle = color;
  wrapText(ctx, text).slice(0, 3).forEach((line, idx, lines) => {
    ctx.fillText(line, canvas.width / 2, canvas.height / 2 + (idx - (lines.length - 1) / 2) * 44);
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

function createSphericalPatch(boundary, color, opacity = 0.78, rings = 7) {
  if (boundary.length < 3) return null;
  const normals = boundary.map((p) => p.clone().normalize());
  const center = centroidOf(boundary, 1).normalize();
  const vertices = [];
  const push = (v) => vertices.push(v.x, v.y, v.z);
  const ringPoint = (ring, index) => {
    const t = ring / rings;
    return slerpUnit(center, normals[index % normals.length], t).multiplyScalar(radius * 1.012);
  };

  for (let i = 0; i < normals.length; i += 1) {
    push(center.clone().multiplyScalar(radius * 1.012));
    push(ringPoint(1, i));
    push(ringPoint(1, i + 1));
  }

  for (let ring = 1; ring < rings; ring += 1) {
    for (let i = 0; i < normals.length; i += 1) {
      const a = ringPoint(ring, i);
      const b = ringPoint(ring, i + 1);
      const c = ringPoint(ring + 1, i);
      const d = ringPoint(ring + 1, i + 1);
      push(a); push(c); push(b);
      push(b); push(c); push(d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color,
      roughness: 0.82,
      metalness: 0.015,
      transparent: true,
      opacity,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2
    })
  );
}

function createBoundaryLine(points, color) {
  if (points.length < 2) return null;
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((p) => p.clone().normalize().multiplyScalar(radius * 1.021)));
  const line = new THREE.LineLoop(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.72 })
  );
  return line;
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
  if (data.type === 'point') return `外层点：${data.point.label}`;
  return '未选择节点';
}

function buildCourse(course, index) {
  const group = new THREE.Group();
  group.position.copy(overviewPosition(index));
  root.add(group);

  const base = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 128, 96),
    new THREE.MeshStandardMaterial({
      color: colorToThree(course.color, 0.62),
      roughness: 0.74,
      metalness: 0.02
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
  group.add(outer, chapters, chapterLabels, sections, sectionLabels, knowledge, knowledgeLabels);

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
    const shade = -0.02 + (chapterIndex % 6) * 0.065;
    const patchColor = colorToThree(course.color, shade).getHex();
    const patch = createSphericalPatch(vectors, patchColor, 0.84, 8);
    if (patch) {
      patch.userData = { type: 'chapter', course, chapter };
      chapters.add(patch);
    }
    const line = createBoundaryLine(vectors, colorToThree(course.color, -0.22).getHex());
    if (line) chapters.add(line);
    if (vectors.length) {
      const label = makeSurfaceLabel(chapter.title, '#111827', 0.96);
      placeSurfaceObject(label, centroidOf(vectors), 1.052);
      label.userData = { type: 'chapter', course, chapter };
      chapterLabels.add(label);
    }
  });

  course.sections.forEach((section, sectionIndex) => {
    const vectors = pathVectors(section.path);
    const shade = 0.12 + (sectionIndex % 5) * 0.045;
    const patch = createSphericalPatch(vectors, colorToThree(course.color, shade).getHex(), 0.76, 6);
    if (patch) {
      patch.userData = { type: 'section', course, section };
      sections.add(patch);
    }
    const line = createBoundaryLine(vectors, colorToThree(course.color, -0.18).getHex());
    if (line) sections.add(line);
    if (vectors.length) {
      const label = makeSurfaceLabel(section.title, '#16202f', 0.68);
      placeSurfaceObject(label, centroidOf(vectors), 1.058);
      label.userData = { type: 'section', course, section };
      sectionLabels.add(label);
    }

    (section.knowledge || []).forEach((point) => {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.11, 14, 10),
        new THREE.MeshStandardMaterial({ color: colorToThree(course.color, -0.3) })
      );
      dot.position.copy(sourceToVector(point, radius * 1.12));
      dot.userData = { type: 'knowledge', course, section, point };
      knowledge.add(dot);

      const label = makeSurfaceLabel(point.label, '#0f172a', 0.35);
      placeSurfaceObject(label, sourceToVector(point), 1.14);
      label.userData = { type: 'knowledge', course, section, point };
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
    title
  });
}

data.courses.forEach(buildCourse);

function setLod(level) {
  lodLabel.textContent = `LOD ${level}`;
  courseGroups.forEach((item) => {
    item.outer.visible = level === 0;
    item.chapters.visible = level === 1;
    item.chapterLabels.visible = level === 1;
    item.sections.visible = level >= 2;
    item.sectionLabels.visible = level === 2;
    item.knowledge.visible = level >= 3;
    item.knowledgeLabels.visible = level >= 3;
    item.title.visible = level === 0;
  });
}

function updateLod() {
  const dist = camera.position.distanceTo(controls.target);
  const level = dist > 82 ? 0 : dist > 43 ? 1 : dist > 25 ? 2 : 3;
  setLod(level);
}

function focusCourse(courseId) {
  selectedCourse = courseGroups.find((item) => item.course.id === courseId) || null;
  courseGroups.forEach((item, index) => {
    item.group.visible = !selectedCourse || item === selectedCourse;
    item.group.userData.target = selectedCourse
      ? new THREE.Vector3(item === selectedCourse ? 0 : (index < 2 ? -82 : 82), 0, selectedCourse && item !== selectedCourse ? -34 : 0)
      : overviewPosition(index);
  });
  if (selectedCourse) {
    courseTitle.textContent = selectedCourse.course.title;
    controls.target.set(0, 0, 0);
    camera.position.set(0, 7, focusDistance);
  } else {
    courseTitle.textContent = '四个教材';
    controls.target.set(0, 0, 0);
    camera.position.set(0, 10, currentOverviewDistance());
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
}

function updateSelectedPoint() {
  if (!selectedNode?.point) return;
  const local = sphericalToVector(Number(phiInput.value), Number(psiInput.value), radius);
  vectorToSource(local, selectedNode.point);
  syncPointObjects(selectedNode.course, selectedNode.point);
}

function setPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
}

function raycastScene(event) {
  setPointer(event);
  const targets = courseGroups.flatMap((item) => [
    item.base,
    item.title,
    ...item.outer.children,
    ...item.chapters.children,
    ...item.chapterLabels.children,
    ...item.sections.children,
    ...item.sectionLabels.children,
    ...item.knowledge.children,
    ...item.knowledgeLabels.children
  ]);
  return raycaster.intersectObjects(targets, false)[0];
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  const hit = raycastScene(event);
  if (!hit) return;
  const item = hit.object.userData;
  if (item?.course) {
    focusCourse(item.course.id);
    selectNode(hit.object);
  }
  if (editMode && item?.point) {
    draggingPoint = { course: item.course, point: item.point };
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
