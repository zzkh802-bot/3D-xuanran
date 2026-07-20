import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  ConfigValidationError,
  createEmptyCourse,
  materializeGeneratedKnowledge,
  normalizeConfig,
  pointFromAngles,
  serializeConfig
} from '../src/config-model.js';
import {
  prepareCourse,
  regionLayout,
  sampleClosedBoundary,
  slerpUnit,
  slerpWithinBoundary,
  sourceToVector,
  sphericalContains,
  vectorToSource
} from '../src/spherical-field.js';
import {
  constrainKnowledgeDirection,
  findClosestSharedBoundary,
  sharedBoundaryNeighbors
} from '../src/editor-constraints.js';

const dataUrl = new URL('../src/data/course-data.json', import.meta.url);
const source = JSON.parse(fs.readFileSync(fileURLToPath(dataUrl), 'utf8'));
const normalized = normalizeConfig(source);

assert.equal(normalized.config.schema, 4);
assert.equal(normalized.config.courses.length, 4);
assert.deepEqual(normalized.warnings, []);
const serialized = serializeConfig(normalized.config);
assert.equal(serialized.includes('"__'), false);
assert.equal(serialized.includes('"generated"'), false);

const emptyConfig = { schema: 4, radius: 10, courses: [] };
const emptyCourse = createEmptyCourse(emptyConfig);
emptyConfig.courses.push(emptyCourse);
assert.equal(normalizeConfig(emptyConfig).config.courses.length, 1);

const vertices = [0, 1, 2].map((index) => (
  pointFromAngles(`vertex-${index + 1}`, `P${index + 1}`, index * 2, 0.35, 10)
));
emptyCourse.vertices.push(...vertices);
emptyCourse.chapters.push({
  id: 'chapter-1',
  title: '测试章节',
  vertexIds: vertices.map((point) => point.id),
  sectionIds: []
});
assert.equal(normalizeConfig(emptyConfig).config.courses[0].chapters.length, 1);

const invalidBoundary = structuredClone(emptyConfig);
invalidBoundary.courses[0].chapters[0].vertexIds = ['vertex-1'];
assert.throws(() => normalizeConfig(invalidBoundary), ConfigValidationError);

const duplicateVertex = structuredClone(emptyConfig);
duplicateVertex.courses[0].vertices[1].id = duplicateVertex.courses[0].vertices[0].id;
assert.throws(() => normalizeConfig(duplicateVertex), ConfigValidationError);

assert.throws(
  () => normalizeConfig({ schema: 4, radius: 0, courses: [] }),
  ConfigValidationError
);
assert.throws(
  () => normalizeConfig({ schema: 999, radius: 10, courses: [] }),
  ConfigValidationError
);
assert.throws(
  () => normalizeConfig({ radius: 10, courses: [] }),
  ConfigValidationError
);

const invalidCoordinate = structuredClone(emptyConfig);
invalidCoordinate.courses[0].vertices[0].phi = null;
assert.throws(() => normalizeConfig(invalidCoordinate), ConfigValidationError);

const runtimeSection = {
  knowledge: [
    { id: 'generated-1', generated: true },
    { id: 'generated-2', generated: true }
  ]
};
assert.equal(materializeGeneratedKnowledge(runtimeSection), true);
assert.equal(runtimeSection.knowledge.every((point) => point.manual && !point.generated), true);

const materializedConfig = structuredClone(normalized.config);
materializedConfig.courses.forEach((course) => {
  prepareCourse(course);
  course.sections.forEach((section) => {
    if (section.knowledge.length) return;
    const layout = regionLayout(course, section);
    const boundary = sampleClosedBoundary(course, section, 6, 1);
    section.knowledge = [0, 1, 2].map((index) => {
      const target = boundary[
        Math.floor(((index + 0.65) / 3) * boundary.length) % boundary.length
      ];
      const direction = target
        ? slerpWithinBoundary(layout.anchor, target, index === 0 ? 0.18 : 0.34, boundary)
        : layout.anchor;
      const point = {
        id: `${section.id}-knowledge-${index + 1}`,
        label: `K${index + 1}`,
        generated: true
      };
      vectorToSource(direction.multiplyScalar(materializedConfig.radius), point, materializedConfig.radius);
      return point;
    });
    materializeGeneratedKnowledge(section);
  });
});
assert.doesNotThrow(() => normalizeConfig(materializedConfig));

const editorConfig = structuredClone(normalized.config);
const editorCourse = editorConfig.courses[0];
prepareCourse(editorCourse);
const editorSection = editorCourse.sections[0];
const editorBoundary = sampleClosedBoundary(editorCourse, editorSection, 8, 1);
const editorStart = regionLayout(editorCourse, editorSection).anchor;
const outsideTarget = editorCourse.sections
  .map((section) => regionLayout(editorCourse, section).anchor)
  .find((direction) => !sphericalContains(direction, editorBoundary))
  || editorStart.clone().negate();
const constrained = constrainKnowledgeDirection(
  editorCourse,
  editorSection,
  editorStart,
  outsideTarget
);
assert.equal(constrained.limited, true);
assert.equal(sphericalContains(constrained.direction, editorBoundary), true);

const neighbors = sharedBoundaryNeighbors(editorSection, editorCourse.sections);
assert.ok(neighbors.length > 0);
const neighbor = neighbors[0];
const neighborEdges = new Set(neighbor.vertexIds.map((startId, index) => (
  [startId, neighbor.vertexIds[(index + 1) % neighbor.vertexIds.length]].sort().join('|')
)));
const sharedEdgeIndex = editorSection.vertexIds.findIndex((startId, index) => (
  neighborEdges.has([
    startId,
    editorSection.vertexIds[(index + 1) % editorSection.vertexIds.length]
  ].sort().join('|'))
));
assert.ok(sharedEdgeIndex >= 0);
const sharedStart = editorCourse.__vertexMap.get(editorSection.vertexIds[sharedEdgeIndex]);
const sharedEnd = editorCourse.__vertexMap.get(
  editorSection.vertexIds[(sharedEdgeIndex + 1) % editorSection.vertexIds.length]
);
const sharedMidpoint = slerpUnit(
  sourceToVector(sharedStart, 1),
  sourceToVector(sharedEnd, 1),
  0.5
);
const sharedMatch = findClosestSharedBoundary(
  editorCourse,
  editorSection,
  editorCourse.sections,
  sharedMidpoint
);
assert.ok(sharedMatch);
assert.ok(sharedMatch.distance < 1e-6);
assert.ok(sharedMatch.regionInsertionIndex > 0);
assert.ok(sharedMatch.neighborInsertionIndex > 0);

console.log('配置模型检查通过：兼容升级、清理、区域约束、共享边界和非法引用校验正常。');
