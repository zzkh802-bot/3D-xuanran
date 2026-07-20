import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  ConfigValidationError,
  normalizeConfig,
  serializeConfig
} from '../src/config-model.js';
import {
  prepareCourse,
  regionBoundaryVectors,
  sourceToVector,
  sphericalContains
} from '../src/spherical-field.js';

const templateUrl = new URL('../course-config.template.json', import.meta.url);
const source = JSON.parse(fs.readFileSync(fileURLToPath(templateUrl), 'utf8'));
const normalized = normalizeConfig(source);

assert.deepEqual(normalized.warnings, []);
assert.equal(normalized.config.schema, 4);
assert.equal(normalized.config.radius, 10);
assert.equal(normalized.config.courses.length, 1);

const course = normalized.config.courses[0];
assert.equal(course.vertices.length, 32);
assert.equal(course.chapters.length, 4);
assert.equal(course.sections.length, 12);
assert.equal(
  course.sections.reduce((sum, section) => sum + section.knowledge.length, 0),
  24
);

const allIds = [
  course.id,
  ...course.vertices.map((vertex) => vertex.id),
  ...course.chapters.map((chapter) => chapter.id),
  ...course.sections.map((section) => section.id),
  ...course.sections.flatMap((section) => section.knowledge.map((point) => point.id))
];
assert.equal(new Set(allIds).size, allIds.length, '模板内所有实体 ID 应保持唯一');

const allPoints = [
  ...course.vertices,
  ...course.sections.flatMap((section) => section.knowledge)
];
allPoints.forEach((point) => {
  assert.ok(
    Math.abs(Math.hypot(point.x, point.y, point.z) - normalized.config.radius) < 1e-4,
    `${point.id} 必须位于指定半径的球面上`
  );
});

prepareCourse(course);
course.chapters.forEach((chapter) => {
  const expectedSectionIds = course.sections
    .filter((section) => section.chapterId === chapter.id)
    .map((section) => section.id);
  assert.deepEqual(chapter.sectionIds, expectedSectionIds);
  assert.equal(expectedSectionIds.length, 3);

  const chapterBoundary = regionBoundaryVectors(course, chapter);
  const chapterSections = course.sections.filter(
    (section) => section.chapterId === chapter.id
  );
  chapterSections.forEach((section) => {
    const sectionBoundary = regionBoundaryVectors(course, section);
    section.vertexIds.forEach((id) => {
      const vertex = course.__vertexMap.get(id);
      assert.ok(
        sphericalContains(sourceToVector(vertex, 1), chapterBoundary),
        `${section.id} 的边界点 ${id} 必须位于所属章节内`
      );
    });
    section.knowledge.forEach((point) => {
      const direction = sourceToVector(point, 1);
      assert.ok(
        sphericalContains(direction, sectionBoundary),
        `${point.id} 必须位于所属小节内`
      );
      assert.ok(
        sphericalContains(direction, chapterBoundary),
        `${point.id} 必须位于所属章节内`
      );
    });
  });
});

const serialized = serializeConfig(course.__vertexMap ? normalized.config : source);
assert.equal(serialized.includes('"__'), false);
assert.deepEqual(normalizeConfig(JSON.parse(serialized)).warnings, []);

const invalidReference = structuredClone(source);
invalidReference.courses[0].sections[0].vertexIds[0] = 'missing-vertex';
assert.throws(() => normalizeConfig(invalidReference), ConfigValidationError);

const duplicateId = structuredClone(source);
duplicateId.courses[0].chapters[1].id = duplicateId.courses[0].chapters[0].id;
assert.throws(() => normalizeConfig(duplicateId), ConfigValidationError);

const invalidContainment = structuredClone(source);
const firstSection = invalidContainment.courses[0].sections[0];
const foreignSection = invalidContainment.courses[0].sections.find(
  (section) => section.chapterId !== firstSection.chapterId
);
firstSection.vertexIds = [...foreignSection.vertexIds];
assert.throws(() => normalizeConfig(invalidContainment), ConfigValidationError);

console.log('模板配置检查通过：结构、ID、坐标、引用、球面包含关系和异常配置校验正常。');
