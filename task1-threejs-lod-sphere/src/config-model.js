const DEFAULT_RADIUS = 10;
const DEFAULT_COURSE_COLOR = '#2878d0';

export class ConfigValidationError extends Error {
  constructor(messages) {
    super(messages.join('\n'));
    this.name = 'ConfigValidationError';
    this.messages = messages;
  }
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function rounded(value, digits = 6) {
  return Number(value.toFixed(digits));
}

function stripRuntime(value) {
  if (Array.isArray(value)) return value.map(stripRuntime);
  if (!value || typeof value !== 'object') return value;
  const clean = {};
  Object.entries(value).forEach(([key, child]) => {
    if (!key.startsWith('__')) clean[key] = stripRuntime(child);
  });
  return clean;
}

function normalizePosition(source, radius, path, errors) {
  let phi = finiteNumber(source.phi);
  let psi = finiteNumber(source.psi);
  let x = finiteNumber(source.x);
  let y = finiteNumber(source.y);
  let z = finiteNumber(source.z);

  if (phi === null || psi === null) {
    if (x === null || y === null || z === null) {
      errors.push(`${path} 缺少有效坐标；需要 phi/psi 或 x/y/z。`);
      return { phi: 0, psi: 0, x: radius, y: 0, z: 0 };
    }
    const length = Math.hypot(x, y, z);
    if (length < 1e-8) {
      errors.push(`${path} 不能位于球心。`);
      return { phi: 0, psi: 0, x: radius, y: 0, z: 0 };
    }
    phi = Math.atan2(y, x);
    psi = Math.asin(Math.max(-1, Math.min(1, z / length)));
  }

  phi = ((phi + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
  psi = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, psi));
  const horizontal = Math.cos(psi);
  x = radius * horizontal * Math.cos(phi);
  y = radius * horizontal * Math.sin(phi);
  z = radius * Math.sin(psi);
  return {
    phi: rounded(phi),
    psi: rounded(psi),
    x: rounded(x, 5),
    y: rounded(y, 5),
    z: rounded(z, 5)
  };
}

function uniqueId(preferred, prefix, used, warnings, path, errors = null) {
  const base = String(preferred || '').trim();
  if (base && !used.has(base)) {
    used.add(base);
    return base;
  }
  if (base && used.has(base) && errors) {
    errors.push(`${path} 的 ID“${base}”重复。`);
  }
  let index = used.size + 1;
  let candidate = `${prefix}-${index}`;
  while (used.has(candidate)) {
    index += 1;
    candidate = `${prefix}-${index}`;
  }
  used.add(candidate);
  if (!base) warnings.push(`${path} 的 ID 为空，已改为“${candidate}”。`);
  return candidate;
}

function normalizeVertex(source, radius, id, path, errors) {
  const clean = stripRuntime(source);
  delete clean.generated;
  return {
    ...clean,
    ...normalizePosition(source, radius, path, errors),
    id,
    label: String(source.label || source.title || id)
  };
}

export function normalizeConfig(rawConfig) {
  const errors = [];
  const warnings = [];
  if (!rawConfig || typeof rawConfig !== 'object' || Array.isArray(rawConfig)) {
    throw new ConfigValidationError(['配置根节点必须是 JSON 对象。']);
  }

  const sourceCourses = Array.isArray(rawConfig.courses) ? rawConfig.courses : null;
  if (!sourceCourses) {
    throw new ConfigValidationError(['配置缺少 courses 数组。']);
  }

  const sourceRadius = finiteNumber(rawConfig.radius);
  const radius = sourceRadius === null ? DEFAULT_RADIUS : sourceRadius;
  if (radius <= 0) errors.push('radius 必须大于 0。');
  const courseIds = new Set();
  const courses = sourceCourses.map((sourceCourse, courseIndex) => {
    const coursePath = `courses[${courseIndex}]`;
    const safeCourse = sourceCourse && typeof sourceCourse === 'object' ? sourceCourse : {};
    const courseId = uniqueId(
      safeCourse.id,
      `course-${courseIndex + 1}`,
      courseIds,
      warnings,
      coursePath,
      errors
    );
    const vertexIds = new Set();
    const vertices = (Array.isArray(safeCourse.vertices) ? safeCourse.vertices : []).map((sourceVertex, index) => {
      const safeVertex = sourceVertex && typeof sourceVertex === 'object' ? sourceVertex : {};
      const id = uniqueId(
        safeVertex.id,
        `${courseId}-vertex`,
        vertexIds,
        warnings,
        `${coursePath}.vertices[${index}]`,
        errors
      );
      return normalizeVertex(
        safeVertex,
        radius,
        id,
        `${coursePath}.vertices[${index}]`,
        errors
      );
    });

    const chapterIds = new Set();
    const chapters = (Array.isArray(safeCourse.chapters) ? safeCourse.chapters : []).map((sourceChapter, index) => {
      const safeChapter = sourceChapter && typeof sourceChapter === 'object' ? sourceChapter : {};
      const id = uniqueId(
        safeChapter.id,
        `${courseId}-chapter`,
        chapterIds,
        warnings,
        `${coursePath}.chapters[${index}]`,
        errors
      );
      const clean = stripRuntime(safeChapter);
      return {
        ...clean,
        id,
        title: String(safeChapter.title || `第 ${index + 1} 章`),
        vertexIds: Array.isArray(safeChapter.vertexIds)
          ? safeChapter.vertexIds.map(String)
          : [],
        sectionIds: []
      };
    });

    const sectionIds = new Set();
    const knowledgeIds = new Set();
    const sections = (Array.isArray(safeCourse.sections) ? safeCourse.sections : []).map((sourceSection, index) => {
      const safeSection = sourceSection && typeof sourceSection === 'object' ? sourceSection : {};
      const id = uniqueId(
        safeSection.id,
        `${courseId}-section`,
        sectionIds,
        warnings,
        `${coursePath}.sections[${index}]`,
        errors
      );
      const clean = stripRuntime(safeSection);
      const knowledge = (Array.isArray(safeSection.knowledge) ? safeSection.knowledge : [])
        .filter((item) => !item?.generated)
        .map((sourcePoint, pointIndex) => {
          const safePoint = sourcePoint && typeof sourcePoint === 'object' ? sourcePoint : {};
          const pointId = uniqueId(
            safePoint.id,
            `${id}-knowledge`,
            knowledgeIds,
            warnings,
            `${coursePath}.sections[${index}].knowledge[${pointIndex}]`,
            errors
          );
          return {
            ...normalizeVertex(
              safePoint,
              radius,
              pointId,
              `${coursePath}.sections[${index}].knowledge[${pointIndex}]`,
              errors
            ),
            manual: true
          };
        });
      return {
        ...clean,
        id,
        title: String(safeSection.title || `第 ${index + 1} 节`),
        chapterId: String(safeSection.chapterId || ''),
        vertexIds: Array.isArray(safeSection.vertexIds)
          ? safeSection.vertexIds.map(String)
          : [],
        knowledge
      };
    });

    chapters.forEach((chapter, chapterIndex) => {
      const path = `${coursePath}.chapters[${chapterIndex}]`;
      const validUnique = new Set(chapter.vertexIds.filter((id) => vertexIds.has(id)));
      chapter.vertexIds.forEach((id) => {
        if (!vertexIds.has(id)) errors.push(`${path} 引用了不存在的顶点“${id}”。`);
      });
      if (validUnique.size < 3) errors.push(`${path} 的边界至少需要 3 个有效且不同的顶点。`);
    });

    sections.forEach((section, sectionIndex) => {
      const path = `${coursePath}.sections[${sectionIndex}]`;
      const chapter = chapters.find((item) => item.id === section.chapterId);
      if (!chapter) errors.push(`${path} 引用了不存在的章节“${section.chapterId || '(空)'}”。`);
      else chapter.sectionIds.push(section.id);
      const validUnique = new Set(section.vertexIds.filter((id) => vertexIds.has(id)));
      section.vertexIds.forEach((id) => {
        if (!vertexIds.has(id)) errors.push(`${path} 引用了不存在的顶点“${id}”。`);
      });
      if (validUnique.size < 3) errors.push(`${path} 的边界至少需要 3 个有效且不同的顶点。`);
    });

    const cleanCourse = stripRuntime(safeCourse);
    return {
      ...cleanCourse,
      id: courseId,
      title: String(safeCourse.title || `课程 ${courseIndex + 1}`),
      color: String(safeCourse.color || DEFAULT_COURSE_COLOR),
      vertices,
      chapters,
      sections
    };
  });

  if (errors.length) throw new ConfigValidationError(errors);
  return {
    config: {
      ...stripRuntime(rawConfig),
      schema: 4,
      radius,
      courses
    },
    warnings
  };
}

export function exportConfig(config) {
  return normalizeConfig(config).config;
}

export function serializeConfig(config, space = 2) {
  return JSON.stringify(exportConfig(config), null, space);
}

export function replaceConfig(target, nextConfig) {
  Object.keys(target).forEach((key) => {
    delete target[key];
  });
  Object.assign(target, nextConfig);
  return target;
}

export function createEmptyCourse(config) {
  const used = new Set(config.courses.map((course) => course.id));
  const id = uniqueId('', 'course', used, [], 'course');
  return {
    id,
    title: `新课程 ${config.courses.length + 1}`,
    color: DEFAULT_COURSE_COLOR,
    vertices: [],
    chapters: [],
    sections: []
  };
}

export function createEntityId(course, type) {
  const collections = {
    vertex: course.vertices,
    chapter: course.chapters,
    section: course.sections,
    knowledge: course.sections.flatMap((section) => section.knowledge || [])
  };
  const used = new Set((collections[type] || []).map((item) => item.id));
  return uniqueId('', `${course.id}-${type}`, used, [], type);
}

export function pointFromAngles(id, label, phi, psi, radius) {
  return {
    id,
    label,
    ...normalizePosition({ phi, psi }, radius, label, []),
    manual: true
  };
}
