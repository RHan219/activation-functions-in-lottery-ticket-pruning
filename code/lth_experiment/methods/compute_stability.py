import argparse
import torch
from pathlib import Path
import json


def load_state(path):
    return torch.load(path, map_location="cpu")


def compute_weight_distance(dense, pruned):
    distances = {}
    for name in dense:
        if name in pruned and dense[name].shape == pruned[name].shape:
            d = torch.norm(dense[name] - pruned[name]) / torch.norm(dense[name])
            distances[name] = d.item()
    return distances


def compute_mask_overlap(mask):
    # mask is a dict of tensors (0/1)
    flat = torch.cat([m.flatten() for m in mask.values()])
    sparsity = 1.0 - flat.mean().item()
    return {"sparsity": sparsity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.path)

    dense_path = run_dir / "dense_reference.pth"
    pruned_path = run_dir / "best_pruned.pth"
    mask_path = run_dir / "mask.pt"

    if not dense_path.exists() or not pruned_path.exists() or not mask_path.exists():
        print(f"Missing files in {run_dir}, skipping.")
        return

    dense = load_state(dense_path)
    pruned = load_state(pruned_path)
    mask = load_state(mask_path)

    distances = compute_weight_distance(dense, pruned)
    mask_info = compute_mask_overlap(mask)

    out = {
        "weight_distance": distances,
        "mask_info": mask_info,
    }

    with open(run_dir / "stability.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved stability metrics to {run_dir / 'stability.json'}")


if __name__ == "__main__":
    main()
