import {
  sampleClosedBoundary,
  slerpUnit,
  slerpWithinBoundary,
  sourceToVector,
  sphericalContains
} from './spherical-field.js';

function regionEdges(region) {
  const vertexIds = region?.vertexIds || [];
  return vertexIds.map((startId, index) => {
    const endId = vertexIds[(index + 1) % vertexIds.length];
    return {
      startId,
      endId,
      key: [startId, endId].sort().join('|'),
      insertionIndex: index + 1
    };
  });
}

export function sharedBoundaryNeighbors(region, candidates) {
  const ownEdges = new Set(regionEdges(region).map((edge) => edge.key));
  return candidates.filter((candidate) => (
    candidate !== region
    && regionEdges(candidate).some((edge) => ownEdges.has(edge.key))
  ));
}

export function findClosestSharedBoundary(course, region, candidates, direction, steps = 36) {
  const ownEdges = regionEdges(region);
  const neighbors = sharedBoundaryNeighbors(region, candidates);
  let best = null;

  neighbors.forEach((neighbor) => {
    const neighborEdges = new Map(regionEdges(neighbor).map((edge) => [edge.key, edge]));
    ownEdges.forEach((edge) => {
      const neighborEdge = neighborEdges.get(edge.key);
      if (!neighborEdge) return;
      const startPoint = course.__vertexMap.get(edge.startId);
      const endPoint = course.__vertexMap.get(edge.endId);
      if (!startPoint || !endPoint) return;
      const start = sourceToVector(startPoint, 1);
      const end = sourceToVector(endPoint, 1);
      for (let step = 0; step <= steps; step += 1) {
        const snappedDirection = slerpUnit(start, end, step / steps);
        const dot = snappedDirection.dot(direction);
        if (!best || dot > best.dot) {
          best = {
            neighbor,
            direction: snappedDirection,
            dot,
            distance: Math.acos(Math.max(-1, Math.min(1, dot))),
            regionInsertionIndex: edge.insertionIndex,
            neighborInsertionIndex: neighborEdge.insertionIndex
          };
        }
      }
    });
  });

  return best;
}

export function constrainKnowledgeDirection(course, section, currentDirection, targetDirection) {
  const boundary = sampleClosedBoundary(course, section, 8, 1);
  const target = targetDirection.clone().normalize();
  if (boundary.length < 3 || sphericalContains(target, boundary)) {
    return { direction: target, limited: false };
  }

  const current = currentDirection.clone().normalize();
  if (!sphericalContains(current, boundary)) {
    return { direction: current, limited: true };
  }

  return {
    direction: slerpWithinBoundary(current, target, 1, boundary),
    limited: true
  };
}
