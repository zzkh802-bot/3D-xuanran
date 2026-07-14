from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent
EXAMPLE_DIR = ROOT / "examples" / "additive_manufacturing" / "sintering_physics"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLE_DIR))

from physicsnemo.models.vfgn.graph_network_modules import VFGNLearnedSimulator  # noqa: E402
from utils import Stats  # noqa: E402


def knn_edges(points: torch.Tensor, k: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points.float(), points.float())
    nearest = torch.topk(distances, k=min(k + 1, points.shape[0]), largest=False).indices[:, 1:]
    senders = torch.arange(points.shape[0]).repeat_interleave(nearest.shape[1])
    receivers = nearest.reshape(-1)
    return senders.long(), receivers.long()


def load_stl_points(stl_path: Path, max_points: int = 64) -> torch.Tensor:
    mesh = trimesh.load_mesh(stl_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] > max_points:
        indices = np.linspace(0, vertices.shape[0] - 1, max_points, dtype=int)
        vertices = vertices[indices]
    center = vertices.mean(axis=0, keepdims=True)
    vertices = vertices - center
    scale = max(np.linalg.norm(vertices, axis=1).max(), 1e-9)
    vertices = vertices / scale * 0.04
    vertices = vertices - vertices.min(axis=0, keepdims=True) + 0.005
    return torch.tensor(vertices, dtype=torch.float64)


def main() -> None:
    torch.manual_seed(11)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stl_path = ROOT / "sample_data" / "stl_foot.stl"
    points = load_stl_points(stl_path, max_points=64)
    num_nodes = points.shape[0]
    seq_len = 5
    pred_len = 1

    steps = []
    for step in range(seq_len):
        shrink = 1.0 - 0.012 * step
        sag = torch.tensor([0.0, 0.0, -0.00025 * step], dtype=torch.float64)
        steps.append(points * shrink + sag)
    positions = torch.stack(steps, dim=1)

    senders, receivers = knn_edges(points, k=6)
    n_particles = torch.tensor([num_nodes], dtype=torch.long)
    n_edges = torch.tensor([senders.numel()], dtype=torch.long)
    particle_types = torch.full((num_nodes,), 2, dtype=torch.long)
    particle_types[0] = 0
    particle_types[1 : min(5, num_nodes)] = 1

    zeros3 = torch.zeros(1, 3, dtype=torch.float64)
    ones3 = torch.ones(1, 3, dtype=torch.float64)
    stats = {
        "velocity": Stats(zeros3, ones3),
        "acceleration": Stats(zeros3, ones3),
        "context": Stats(torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)),
    }

    model = VFGNLearnedSimulator(
        num_dimensions=3 * pred_len,
        num_seq=seq_len,
        boundaries=torch.tensor([[0.0, 0.06], [0.0, 0.06], [0.0, 0.06]], dtype=torch.float64),
        num_particle_types=3,
        particle_type_embedding_size=16,
        normalization_stats=stats,
        connectivity_param=0.06,
    ).to(device)
    model.setMessagePassingDevices([device])
    if device != "cpu":
        model._graph_network.set_device([device])
    model.eval()

    with torch.no_grad():
        out = model.inference(
            position_sequence=positions.to(device),
            n_particles_per_example=n_particles.to(device),
            n_edges_per_example=n_edges.to(device),
            senders=senders.to(device),
            receivers=receivers.to(device),
            predict_length=pred_len,
            global_context=None,
            particle_types=particle_types.to(device),
        )

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "stl_foot_vfgn_smoke_prediction.npy", out.detach().cpu().numpy())
    print("stl:", stl_path)
    print("device:", device)
    print("sampled_nodes:", num_nodes)
    print("edges:", senders.numel())
    print("input_position_sequence:", tuple(positions.shape))
    print("predicted_next_positions:", tuple(out.shape))
    print("saved:", output_dir / "stl_foot_vfgn_smoke_prediction.npy")
    print("sample_prediction:", out[0, 0].detach().cpu().numpy().round(8).tolist())


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
