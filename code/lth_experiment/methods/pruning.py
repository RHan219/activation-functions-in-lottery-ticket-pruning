import torch.nn as nn
import torch

def get_weight_tensors(model):
    return [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

def magnitude_prune(model, sparsity: float):
    params = get_weight_tensors(model)
    all_weights = torch.cat([p.data.view(-1).abs() for p in params])
    k = int((1 - sparsity) * all_weights.numel())
    if k == 0:
        return

    threshold, _ = torch.kthvalue(all_weights, k)
    for p in params:
        mask = (p.data.abs() >= threshold).float()
        p.data.mul_(mask)

def grasp_score(model, loss_fn, data_loader, device, num_batches=1, temp=200.0):
    model.eval()
    # zero gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    it = iter(data_loader)
    for _ in range(num_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = loss_fn(outputs, y) / temp
        # retain graph for multiple backward calls if needed
        loss.backward()

    # compute GraSP scores: -w * g^2
    scores = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            scores[name] = -(p * p.grad * p.grad)
    return scores

def apply_grasp_pruning(model, scores, sparsity):
    # flatten all scores
    all_scores = torch.cat([s.flatten() for s in scores.values()])
    k = int((1 - sparsity) * all_scores.numel())
    if k <= 0:
        threshold = torch.max(all_scores) + 1  # masks everything
    else:
        threshold, _ = torch.kthvalue(all_scores, k)

    # build mask dict
    mask = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in scores:
                m = (scores[name] >= threshold).float()
                p.mul_(m)
                mask[name] = m
    return mask
