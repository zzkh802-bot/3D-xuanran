from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
EXAMPLE_DIR = ROOT / "examples" / "additive_manufacturing" / "sintering_physics"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLE_DIR))

from physicsnemo.models.vfgn.graph_network_modules import VFGNLearnedSimulator  # noqa: E402
from utils import Stats  # noqa: E402


def fully_connected_edges(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    senders = []
    receivers = []
    for sender in range(num_nodes):
        for receiver in range(num_nodes):
            if sender == receiver:
                continue
            senders.append(sender)
            receivers.append(receiver)
    return torch.tensor(senders, dtype=torch.long), torch.tensor(receivers, dtype=torch.long)


def main() -> None:
    torch.manual_seed(7)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_nodes = 8
    seq_len = 5
    dim = 3
    pred_len = 1

    base = torch.rand(num_nodes, dim, dtype=torch.float64) * 0.02
    drift = torch.tensor([0.0008, -0.0002, 0.0003], dtype=torch.float64)
    positions = torch.stack([base + drift * step for step in range(seq_len)], dim=1)
    particle_types = torch.full((num_nodes,), 2, dtype=torch.long)
    particle_types[0] = 0
    particle_types[1] = 1

    senders, receivers = fully_connected_edges(num_nodes)
    n_particles = torch.tensor([num_nodes], dtype=torch.long)
    n_edges = torch.tensor([senders.numel()], dtype=torch.long)

    zeros3 = torch.zeros(1, dim, dtype=torch.float64)
    ones3 = torch.ones(1, dim, dtype=torch.float64)
    stats = {
        "velocity": Stats(zeros3, ones3),
        "acceleration": Stats(zeros3, ones3),
        "context": Stats(torch.zeros(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)),
    }

    model = VFGNLearnedSimulator(
        num_dimensions=dim * pred_len,
        num_seq=seq_len,
        boundaries=torch.tensor([[0.0, 0.05], [0.0, 0.05], [0.0, 0.05]], dtype=torch.float64),
        num_particle_types=3,
        particle_type_embedding_size=16,
        normalization_stats=stats,
        connectivity_param=0.05,
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

    print("device:", device)
    print("input_position_sequence:", tuple(positions.shape))
    print("edges:", senders.numel())
    print("predicted_next_positions:", tuple(out.shape))
    print("sample_prediction:", out[0, 0].detach().cpu().numpy().round(8).tolist())


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
