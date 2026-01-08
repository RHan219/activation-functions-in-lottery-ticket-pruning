# Results and Analysis of Project Experiments

## Experiment

For my experiments, my main objective was to evaluate the impact that different activation
functions have on pruning effectiveness, as well as the effect they had on the stability of the
models that were trained using the lottery ticket hypothesis as its base.

For the project, I used a dense, shallow CNN trained on CIFAR-10. The three activation
functions that I used were: ReLU, tanh, and sigmoid.

When pruning the models, I used two different pruning methods:

**Magnitude pruning**, which ranks weights by absolute value and retains the top \( k \) impactful
weights. This was determined using binary masks on the weights and the function:

$$m_i = \mathbf{1}(|w_i| \ge \tau_k)$$

where \( w_i \) is the value of the weight, and \( \tau_k \) is the weight value of the \( k \)-th weight.
(This allows us to keep the top \( k \) important/largest weights.)

**GraSP pruning**, which scores weights by their contribution to gradient flow, using the formula:

$$\text{score}_i = -w_i \cdot g_i^2$$

where \( g_i \) is the gradient of the loss with respect to the weight \( w_i \).

While our function for GraSP score may appear off, GraSP scores are designed to preserve
gradient flow after pruning. While the original formula involves second-order derivatives (the
Hessian Matrix), due to restricted setup, we instead adopt a first-order approximation for
computational efficiency, using the gradient instead.

Pruning masks are applied to the initial weights, and pruned models are retrained from scratch.
We evaluate performance across sparsity levels \( s = \{0.5, 0.8, 0.9\} \); meaning we check after
50%, 80%, and 90% of the weights have been pruned.

Each model is trained for 50 epochs and pruned for 20 epochs.

---

## Results and Analysis

For our results, we have 4 different sets of graphs.

---

## Validation Accuracy vs. Sparsity

![Accuracy vs Sparsity](images/image11.png)

This graph tells us about the validation accuracy of each of our pruned models at each of our
checkpointed sparsity levels (50%, 80%, 90%).

We can see that for the **ReLU** activation function, the magnitude pruning peaks at 0.8 sparsity,
only to drop by a decent margin at 0.9 sparsity. Meanwhile, the accuracy with GraSP is
unchanging, and stays consistent across all sparsities.

For **Tanh**, we can see that both magnitude pruning and GraSP pruning had relatively equal
accuracy results, however, the accuracy for GraSP pruning was slightly lower than that of
magnitude pruning until 0.9 sparsity. We can also see that the Tanh models both had lower
accuracies than the two ReLU models.

For **Sigmoid**, the GraSP pruning greatly outperformed the magnitude pruning; this was likely
caused by oversaturating due to the highly volatile nature of sigmoid's gradients.

---

## Mask Sparsity vs Target Sparsity

![Mask Sparsity](images/image12.png)

This graph shows how close the actual sparsity of the pruning mask is to the intended target
sparsity.

For **ReLU**, GraSP pruning failed to prune the model, as the actual sparsity remained around
0.05 regardless of the target sparsity. Magnitude pruning, however, was able to prune the model
correctly.

For **Tanh** and **Sigmoid**, both magnitude pruning and GraSP pruning were able to prune the
models correctly, with the actual sparsity matching the target sparsity.

---

## Stability vs Sparsity

![Stability vs Sparsity](images/image13.png)

This graph shows the stability of the pruned models at each sparsity level. Stability is measured
using the following equation:

$$D = \frac{\|W_{\text{dense}} - W_{\text{pruned}}\|_2}{\|W_{\text{dense}}\|_2}$$

For **ReLU**, both magnitude pruning and GraSP pruning had similar stability values, with both
remaining relatively stable across all sparsity levels.

For **Tanh**, both pruning methods again had similar stability values, with both remaining stable
across all sparsity levels.

For **Sigmoid**, magnitude pruning remained stable, however, GraSP pruning had a large spike
in instability at 0.8 sparsity. This was likely caused by the highly volatile nature of sigmoid's
gradients.

---

## Training Curves — ReLU

![Training Curves — ReLU](images/image14.png)

This graph shows the training accuracy of the dense model, the magnitude-pruned model, and
the GraSP-pruned model for the ReLU activation function.

Both pruned models initially outperform the dense model in terms of training accuracy,
especially in the early epochs. However, the dense model eventually catches up and surpasses
the pruned models in later epochs.

---

## Training Curves — Tanh

![Training Curves — Tanh](images/image15.png)

This graph shows the training accuracy of the dense model, the magnitude-pruned model, and
the GraSP-pruned model for the Tanh activation function.

All three models start with similar accuracy and improve significantly around epoch 10. The
dense model achieves the highest training accuracy overall, while the pruned models show
slightly lower performance.

---

## Training Curves — Sigmoid

![Training Curves — Sigmoid](images/image16.png)

This graph shows the training accuracy of the dense model, the magnitude-pruned model, and
the GraSP-pruned model for the Sigmoid activation function.

The dense and magnitude-pruned models both fail to learn, with accuracy remaining around
0.10. Meanwhile, the GraSP-pruned model shows a significant increase in accuracy, reaching
above 0.35 by epoch 20 and then plateauing.

