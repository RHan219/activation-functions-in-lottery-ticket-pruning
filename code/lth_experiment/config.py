import torch

CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,

    # Data
    "batch_size": 64,
    "num_workers": 0,
    "grasp_batches": 1,   # number of batches to compute GraSP scores

    # Training
    "num_epochs_dense": 50,
    "num_epochs_pruned": 20,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_schedule_milestones": [10, 15],
    "lr_schedule_gamma": 0.1,

    # Paths
    "data_dir": "./data",
    "checkpoint_dir": "./checkpoints",
    "results_dir": "./results",

    # Pruning
    "sparsity_levels": [0.5, 0.8, 0.9],

    # Activations
    "activations": ["relu", "tanh", "sigmoid"],
}
