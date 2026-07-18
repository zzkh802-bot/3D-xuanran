import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { pointFromAngles } from '../src/config-model.js';

const radius = 10;
const outputUrl = new URL('../course-config.template.json', import.meta.url);
const latitudeLevels = [-0.9, -0.3, 0.3, 0.9];
const chapterSpecs = [
  {
    title: '第一章 课程基础',
    description: '建立课程全局认识，准备学习工具并掌握基本概念。',
    phiRange: [-2.95, -1.65],
    sections: [
      ['1.1 课程导论', '了解课程目标、内容结构与学习路线。', ['课程目标', '学习路线']],
      ['1.2 基本概念', '建立后续学习所需的核心术语体系。', ['核心术语', '概念关系']],
      ['1.3 工具与环境', '完成学习及实践环境的准备工作。', ['工具选择', '环境配置']]
    ]
  },
  {
    title: '第二章 核心理论',
    description: '理解课程的基本原理、模型结构和分析方法。',
    phiRange: [-1.45, -0.15],
    sections: [
      ['2.1 基本原理', '学习支撑课程体系的主要规律。', ['关键定律', '适用条件']],
      ['2.2 模型结构', '理解从问题抽象到模型表达的过程。', ['模型要素', '结构关系']],
      ['2.3 分析方法', '掌握常用分析方法及其选择依据。', ['方法分类', '选择策略']]
    ]
  },
  {
    title: '第三章 实践应用',
    description: '通过案例、实验和评价环节应用核心理论。',
    phiRange: [0.15, 1.45],
    sections: [
      ['3.1 案例分析', '将理论用于典型案例的拆解与解释。', ['问题识别', '案例推演']],
      ['3.2 实验设计', '设计可执行、可复现的验证过程。', ['变量控制', '实验步骤']],
      ['3.3 结果评价', '使用合适指标分析实验和案例结果。', ['评价指标', '误差分析']]
    ]
  },
  {
    title: '第四章 综合提升',
    description: '完成综合项目，并形成总结、复盘与拓展能力。',
    phiRange: [1.65, 2.95],
    sections: [
      ['4.1 项目设计', '明确综合项目的目标、范围和实施方案。', ['需求分析', '方案设计']],
      ['4.2 项目实现', '按计划完成实现、验证和迭代。', ['任务实施', '质量验证']],
      ['4.3 总结拓展', '沉淀成果并探索进一步学习方向。', ['成果复盘', '拓展方向']]
    ]
  }
];

function vertexId(chapterNumber, levelNumber, side) {
  return `template-c${chapterNumber}-level-${levelNumber}-${side}`;
}

function makePoint(id, label, phi, psi) {
  return pointFromAngles(id, label, phi, psi, radius);
}

const vertices = [];
const chapters = [];
const sections = [];

chapterSpecs.forEach((chapterSpec, chapterIndex) => {
  const chapterNumber = chapterIndex + 1;
  const [leftPhi, rightPhi] = chapterSpec.phiRange;

  latitudeLevels.forEach((psi, levelIndex) => {
    const levelNumber = levelIndex + 1;
    vertices.push(
      makePoint(
        vertexId(chapterNumber, levelNumber, 'left'),
        `第${chapterNumber}章 L${levelNumber} 左边界点`,
        leftPhi,
        psi
      ),
      makePoint(
        vertexId(chapterNumber, levelNumber, 'right'),
        `第${chapterNumber}章 L${levelNumber} 右边界点`,
        rightPhi,
        psi
      )
    );
  });

  const chapterId = `template-chapter-${chapterNumber}`;
  const sectionIds = chapterSpec.sections.map(
    (_, sectionIndex) => `template-section-${chapterNumber}-${sectionIndex + 1}`
  );
  chapters.push({
    id: chapterId,
    title: chapterSpec.title,
    description: chapterSpec.description,
    vertexIds: [
      vertexId(chapterNumber, 1, 'left'),
      vertexId(chapterNumber, 1, 'right'),
      vertexId(chapterNumber, 4, 'right'),
      vertexId(chapterNumber, 4, 'left')
    ],
    sectionIds
  });

  chapterSpec.sections.forEach((sectionSpec, sectionIndex) => {
    const sectionNumber = sectionIndex + 1;
    const sectionId = `template-section-${chapterNumber}-${sectionNumber}`;
    const lowerPsi = latitudeLevels[sectionIndex];
    const upperPsi = latitudeLevels[sectionIndex + 1];
    const centerPhi = (leftPhi + rightPhi) / 2;
    const centerPsi = (lowerPsi + upperPsi) / 2;
    const [title, description, knowledgeLabels] = sectionSpec;
    const knowledge = knowledgeLabels.map((label, knowledgeIndex) => {
      const direction = knowledgeIndex === 0 ? -1 : 1;
      return {
        ...makePoint(
          `template-knowledge-${chapterNumber}-${sectionNumber}-${knowledgeIndex + 1}`,
          label,
          centerPhi + direction * 0.18,
          centerPsi + direction * 0.045
        ),
        description: `${title}中的“${label}”知识点。`
      };
    });

    sections.push({
      id: sectionId,
      title,
      description,
      chapterId,
      vertexIds: [
        vertexId(chapterNumber, sectionNumber, 'left'),
        vertexId(chapterNumber, sectionNumber, 'right'),
        vertexId(chapterNumber, sectionNumber + 1, 'right'),
        vertexId(chapterNumber, sectionNumber + 1, 'left')
      ],
      knowledge
    });
  });
});

const template = {
  $schema: './course-config.schema.json',
  schema: 4,
  radius,
  courses: [
    {
      id: 'template-course',
      title: '示例课程：从基础到综合实践',
      description: '一份可直接导入和继续编辑的完整课程模板。',
      color: '#2878d0',
      vertices,
      chapters,
      sections
    }
  ]
};

fs.writeFileSync(fileURLToPath(outputUrl), `${JSON.stringify(template, null, 2)}\n`);
console.log(
  `模板已生成：${chapters.length} 章、${sections.length} 节、`
  + `${sections.reduce((sum, section) => sum + section.knowledge.length, 0)} 个知识点。`
);
