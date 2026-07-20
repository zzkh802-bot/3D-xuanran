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
  slerpWithinBoundary,
  vectorToSource
} from '../src/spherical-field.js';

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

console.log('配置模型检查通过：兼容升级、清理、空课程和非法引用校验正常。');
