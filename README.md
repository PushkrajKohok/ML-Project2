## Boosting Trees


## What does the model you have implemented do and when should it be used?

We have implemented a binary classification model using **Gradient Boosting Decision Trees**, trained from first principles with logistic loss. The model combines multiple shallow regression trees where each tree corrects the residuals (pseudo-gradients) of the previous ensemble. The model's predictions are additive in the function space and refined stage-by-stage using an optimized step size \( γm \), improving both accuracy and convergence.


This model is most effective in structured/tabular datasets, especially when feature relationships are non-linear and performance is improved by capturing high-order interactions. It is suitable for binary classification tasks where interpretability (via feature importances and loss monitoring) is also important.

---

## How did you test your model to determine if it is working reasonably correctly?

We employed a comprehensive and multi-layered testing strategy:

1. **Quantitative evaluation** on:
   - Synthetic datasets: moons, circles, linearly separable data
   - Metrics: accuracy, loss progression, gamma value tracking

2. **Visual diagnostics**:
   - Decision boundary plots
   - Training log-loss curves per iteration
   - Gamma value per boosting stage

3. **Structured testing suite** with over 30 test cases:
   - Functional tests (e.g., initialization, learning rate scaling, proba output)
   - Edge cases (NaN handling, all-labels-equal, repeated `.fit()`)
   - Stress and combination tests (deep trees + small leaves, high-dim with max_features)

These tests are located in the `tests/test_gradient_boosting_classifier.py` file and can be run using:
```bash
pytest -v tests/test_gradient_boosting_classifier.py
```

---

## What parameters have you exposed to users of your implementation in order to tune performance?

Our implementation exposes the following hyperparameters:

- `n_estimators`: Number of trees to be trained in the boosting sequence
- `learning_rate`: Scaling factor on each new tree (shrinkage)
- `max_depth`: Maximum depth of individual decision trees
- `min_samples_leaf`: Minimum number of samples in a tree leaf node
- `max_features`: Number of features considered per split (feature subsampling)
- `subsample`: Fraction of training rows used per tree (data bagging)
- `verbose`: Boolean to control detailed printing/logs

### Example Usage:
```python
clf = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=2,
    max_features=1,
    subsample=0.8,
    verbose=False
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes, the model currently has the following known limitations:

1. **Multiclass classification** is not yet supported. The model is designed for binary classification only. Extending it would involve implementing softmax loss and one-vs-rest strategies.

2. **NaN or missing input values** are not automatically handled. This can be mitigated by integrating preprocessing modules such as `SimpleImputer` or by using surrogate splits.

3. **Highly imbalanced class distributions** may affect convergence in early stages, though our initialization using log-odds helps. Future work could include focal loss or cost-sensitive learning.

4. **Single-class labels (e.g., all y = 1)** cause the model to raise a `ValueError`, which is expected behavior. This is a fundamental requirement for supervised learning and cannot be bypassed meaningfully.





#  Gradient Boosting from Scratch 

This project involves the implementation of a binary classification algorithm based on **Gradient Boosting Decision Trees** using only core Python and NumPy. The model is optimized for log-loss and includes multiple advanced features such as optimized gamma calculation, custom decision trees with feature and data subsampling, staged prediction tracking, and thorough visualization for analysis and validation.

---

##  Overview

Gradient Boosting is an ensemble method that builds a sequence of weak learners (typically decision trees) where each new learner corrects the errors of the existing ensemble. Our implementation is focused on binary classification using the **logistic loss function** and supports the following core features:

  ## How It Works

1. **Initialization (log-odds of the prior)**  
   ```text
   F₀(x) = log(p / (1 - p))
   where p = (1/n) · ∑ᵢ yᵢ


- Computation of residuals (negative gradients):
  ```text
  rᵢ⁽ᵐ⁾ = yᵢ – sigmoid(Fₘ₋₁(xᵢ)), where
  sigmoid(z) = 1 / (1 + exp(–z))


- Tree updates scaled by an optimal gamma:
  ```text
  γⱼᵐ = ∑_{i∈Rⱼᵐ} (yᵢ – pᵢ)
      ─────────────────────── , 
      ∑_{i∈Rⱼᵐ} pᵢ (1 – pᵢ)
  where pᵢ = sigmoid(Fₘ₋₁(xᵢ))

- Model update with shrinkage
  ```text
  Fₘ(x) = Fₘ₋₁(x) + ν · ∑ⱼ γⱼᵐ · I{x ∈ Rⱼᵐ}

- Log-loss convergence tracking across iterations
  ```text
  L⁽ᵐ⁾ = – (1/n) · ∑ᵢ [ yᵢ·log(pᵢ) + (1–yᵢ)·log(1–pᵢ) ]
  where pᵢ = sigmoid(Fₘ(xᵢ))

- Visualization of decision boundaries, gamma evolution, feature importances, and loss progression.

The final model is highly configurable, robust, and extensively tested.

---


---

##  Project Objectives

- Implement a Gradient Boosting Classifier from scratch without external ML libraries.
- Support binary classification using the logistic loss function.
- Enable complete configurability via hyperparameters such as learning rate, max depth, subsample ratio, and more.
- Visualize internal model behavior including training loss, gamma scaling, and decision boundaries.
- Rigorously test all aspects of the model with both functional and corner-case test suites.

---

##  Configuration Parameters

The model supports the following key hyperparameters for tuning:

- `n_estimators` — Number of boosting rounds (trees).
- `learning_rate` — Multiplier applied to each tree's output.
- `max_depth` — Maximum depth of each regression tree.
- `min_samples_leaf` — Minimum number of samples per leaf node.
- `max_features` — Number of features considered per split.
- `subsample` — Fraction of training data used per tree.
- `verbose` — Toggles verbose training output.

---

##  Complete Test Case Inventory

###  Original Functional Tests
1. **test_init_defaults** – Tests default hyper-parameters.
2. **test_predict_before_fit** – Ensures predict() cannot be called before fit().
3. **test_zero_estimators** –  Verifies that n_estimators=0 raises a ValueError.

###  Functional Extension Tests
4. **test_learns_problems** –  Checks that the model learns on linearly separable, moons, and circles datasets.
5. **test_learning_robustness** – Parametrized test across learning rates to ensure robust accuracy on moons.
6. **test_n_estimators_increase_score** – Checks that accuracy improves (or stays above threshold) as n_estimators increases.
7. **test_reproducibility** – Verifies identical predictions given the same seed and that training loss is non-increasing and finite.
8. **test_feature_importance_sum** –  Ensures feature importance scores sum to 1 and are non-negative.

###  Corner Case Tests
9. **test_nan_input** –  Confirms that NaNs in X raise a ValueError during fit().
10. **test_single_class_error** – GEnsures that fitting on a single-class target raises a ValueError.
11. **test_near_perfect_collinearity** – Verifies training succeeds with nearly duplicate features.
12. **test_large_n_estimators_runtime** – Stress‑test: checks runtime and accuracy with 300 trees.

###  Additional Tests
13. **test_subsample_robustness** – Ensures model learns even with row subsampling (subsample<1)
14. **test_max_depth_effect** –  Compares shallow vs deep trees to confirm deeper trees outperform by a margin.

---

##  Visual Analysis Components

Our `BoostingTreeVisualization.ipynb` includes the following visualizations:

- Log-loss curves over iterations
- Gamma values per tree
- Decision boundaries (train/test)
- Feature importance bar plots
- Effect of hyperparameters like `min_samples_leaf` and `max_features`

---

##  Enhancements and Technical Strengths

###  Gamma Optimization:
Used analytic derivation for optimal scaling per iteration to ensure fast convergence and minimized loss.

###  Tree-based Feature Importance:
Calculated as the total split-gain per feature, then normalized so that all importances sum to 1:

 ```text
 # 1. Raw importance: sum of gains for feature j across all splits
 importance_j = sum_{splits s where feature = j} gain_s

 # 2. Normalization: divide by total importance
 importance_j = importance_j / sum_{k=1}^p importance_k
 ```
###  Sample and Feature Subsampling:
Implemented `subsample` and `max_features` for bias-variance tradeoffs and improved generalization.

###  Full Training Traceability:
Added `self.errors_`, `self.gammas`, `staged_predict()` — fully transparent model updates.

###  Modular Testing:
Structured for `pytest` compatibility with 32 granular test cases.

---

##  Future Work

- Add early stopping based on validation loss
- Handle missing values using split-aware imputers
- Extend to multiclass classification
- Implement pruning or tree dropout
- Export models as JSON or binary blobs

---

##  Troubleshooting Guide

| Symptom | Solution |
|---------|----------|
| Verbose output flooding console | Set `verbose=False` in model |
| Predicts same class always | Check if all `y` = same label |
| Error on NaNs | Use preprocessing to impute or drop missing rows |
| Crash during `predict()` | Ensure `.fit()` is called first |
| Low accuracy on small datasets | Tune `max_depth` and `min_samples_leaf` |

---
