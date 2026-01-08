import torch
import argparse
from pathlib import Path


def load_state(path):
    """Load a PyTorch state dict from file."""
    return torch.load(path, map_location="cpu")


def filter_weights(state):
    """Return only Conv2d/Linear weights from a state dict."""
    return {k: v for k, v in state.items() if "weight" in k}


def weight_space_stability(dense_state, pruned_state, eps=1e-8):
    """
    Compute normalized L2 distance between dense and pruned weights.
    """
    dense_w = filter_weights(dense_state)
    pruned_w = filter_weights(pruned_state)

    assert dense_w.keys() == pruned_w.keys(), "State dicts must match."

    dists = []
    for name in dense_w.keys():
        w_dense = dense_w[name].float()
        w_pruned = pruned_w[name].float()

        diff = torch.norm(w_dense - w_pruned).item()
        denom = torch.norm(w_dense).item() + eps
        dists.append(diff / denom)

    return sum(dists) / len(dists)


def mask_overlap(mask_dense, mask_pruned):
    """
    Compute overlap between two binary masks.
    """
    total_kept = 0
    total_overlap = 0

    for name in mask_pruned.keys():
        m_p = mask_pruned[name]
        m_d = mask_dense[name]

        kept = m_p.sum().item()
        overlap = (m_p * m_d).sum().item()

        total_kept += kept
        total_overlap += overlap

    if total_kept == 0:
        return 0.0

    return total_overlap / total_kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Path to activation/sparsity folder (e.g., results/relu/prune_0.8)")
    args = parser.parse_args()

    folder = Path(args.path)

    dense_path = folder / "dense_reference.pth"
    pruned_path = folder / "pruned_final.pth"
    mask_path = folder / "mask.pt"

    print(f"\n=== Computing stability for: {folder} ===")

    dense_state = load_state(dense_path)
    pruned_state = load_state(pruned_path)

    # Weight-space stability
    ws = weight_space_stability(dense_state, pruned_state)
    print(f"Weight-space stability: {ws:.6f}")

    # Mask-overlap stability (optional)
    if mask_path.exists():
        mask_pruned = load_state(mask_path)
        # Dense mask is all ones (dense model has no zeros)
        mask_dense = {k: torch.ones_like(v) for k, v in mask_pruned.items()}
        mo = mask_overlap(mask_dense, mask_pruned)
        print(f"Mask-overlap stability: {mo:.6f}")
    else:
        print("No mask.pt found — skipping mask-overlap stability.")


if __name__ == "__main__":
    main()
