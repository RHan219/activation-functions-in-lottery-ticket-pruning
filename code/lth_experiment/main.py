import argparse
import torch
from pathlib import Path

from config import CONFIG
from activations import ACTIVATIONS
from models.shallow_cnn import ShallowCNN
from utils.load_data import get_cifar10_loaders
from utils.train import train_one_epoch, evaluate, set_seed
from methods.pruning import magnitude_prune, grasp_score, apply_grasp_pruning
from utils.log_data import DataLogger


def train_dense(activation_name):
    """Train a dense model for one activation and return initial + final states."""
    set_seed(CONFIG["seed"])

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        subset_size=CONFIG["subset_size_quick"]
    )

    activation_fn = ACTIVATIONS[activation_name]
    model = ShallowCNN(activation_fn=activation_fn).to(CONFIG["device"])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=CONFIG["lr_schedule_milestones"],
        gamma=CONFIG["lr_schedule_gamma"]
    )

    logger = DataLogger(CONFIG["results_dir"], activation_name, "dense")
    logger.log_config(CONFIG)

    best_acc = 0.0
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_state = None

    for epoch in range(1, CONFIG["num_epochs_dense"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"]
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, CONFIG["device"]
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)

        # Track best dense model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.save_model(best_state, "best_dense.pth")
            logger.save_model(initial_state, "initial_weights.pth")

    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        logger.save_model(best_state, "best_dense.pth")
        logger.save_model(initial_state, "initial_weights.pth")

    final_dict = {"best_val_acc": best_acc}

    logger.log_final(final_dict)
    return initial_state, best_state


def run_pruning(activation_name, initial_state, dense_state, sparsity, prune_method="magnitude"):
    """Run magnitude pruning + rewinding + retraining."""
    set_seed(CONFIG["seed"])

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        subset_size=CONFIG["subset_size_quick"]
    )

    activation_fn = ACTIVATIONS[activation_name]
    model = ShallowCNN(activation_fn=activation_fn).to(CONFIG["device"])
    model.load_state_dict(dense_state)

    # Set the desired pruning method
    if prune_method == "magnitude":
        magnitude_prune(model, sparsity)
        mask = {name: (p != 0).float() for name, p in model.state_dict().items()}
        prune_tag = f"prune_mag_{sparsity}"

    elif prune_method == "grasp":
        # small loader for GraSP scoring
        small_loader, _ = get_cifar10_loaders(
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            subset_size=CONFIG["subset_size_quick"]
        )
        criterion = torch.nn.CrossEntropyLoss()

        scores = grasp_score(
            model,
            criterion,
            small_loader,
            CONFIG["device"],
            num_batches=CONFIG.get("grasp_batches", 1)
        )

        mask = apply_grasp_pruning(model, scores, sparsity)
        prune_tag = f"prune_grasp_{sparsity}"

    else:
        raise ValueError(f"Unknown prune method: {prune_method}")


    logger = DataLogger(CONFIG["results_dir"], activation_name, prune_tag)
    logger.log_config(CONFIG)
    logger.save_mask(mask, "mask.pt")

    logger.save_model(dense_state, "dense_reference.pth")

    # Rewind surviving weights to initial values
    pruned_state = model.state_dict()
    rewound_state = {}

    for name, w_pruned in pruned_state.items():
        w_init = initial_state[name]
        if w_pruned.shape == w_init.shape:
            rewound_state[name] = w_init * (w_pruned != 0)
        else:
            rewound_state[name] = w_init

    model.load_state_dict(rewound_state)

    # Retrain pruned model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=CONFIG["lr_schedule_milestones"],
        gamma=CONFIG["lr_schedule_gamma"]
    )

    best_acc = 0.0
    best_pruned_state = None

    for epoch in range(1, CONFIG["num_epochs_pruned"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"]
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, CONFIG["device"]
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr, sparsity)

        if val_acc > best_acc:
            best_acc = val_acc
            best_pruned_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.save_model(best_pruned_state, "best_pruned.pth")

    if best_pruned_state is None:
        best_pruned_state = {k: v.clone() for k, v in model.state_dict().items()}
        logger.save_model(best_pruned_state, "best_pruned.pth")

    # Store pruned final weights
    logger.save_model(best_pruned_state, "pruned_final.pth")

    logger.log_final({
        "best_val_acc": best_acc,
        "sparsity": sparsity,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training + pruning pipeline")
    parser.add_argument("--analyze", action="store_true", help="Run analysis only")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    if args.train:
        print("\n=== FULL TRAINING PIPELINE ===")
        for activation_name in CONFIG["activations"]:
            print(f"\n=== Training dense model for {activation_name} ===")
            initial_state, dense_state = train_dense(activation_name)

            for sparsity in CONFIG["sparsity_levels"]:
                print(f"--- Magnitude pruning {activation_name} at sparsity {sparsity} ---")
                run_pruning(activation_name, initial_state, dense_state, sparsity, prune_method="magnitude")

                print(f"--- GraSP pruning {activation_name} at sparsity {sparsity} ---")
                run_pruning(activation_name, initial_state, dense_state, sparsity, prune_method="grasp")

        print("\nTraining + pruning completed")
        return

    if args.analyze:
        print("\n=== FULL ANALYSIS PIPELINE ===")

        results_root = Path(CONFIG["results_dir"])

        for activation_dir in results_root.iterdir():
            if not activation_dir.is_dir():
                continue

            for prune_dir in activation_dir.iterdir():
                if prune_dir.is_dir() and prune_dir.name.startswith("prune_"):
                    print(f"\nRunning analysis on: {prune_dir}")
                    import subprocess
                    subprocess.run(["python", "methods/compute_stability.py", "--path", str(prune_dir)])

        print("\nFull analysis completed")
        return

    print("Please specify --train or --analyze")


if __name__ == "__main__":
    main()
