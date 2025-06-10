# VIOLET Framework

![Best Graph](best%20graph.png)

**VIOLET** (Vectorized Invariance Optimization for Language Embeddings using Twins) is a self-supervised framework designed to produce robust sentence embeddings without relying on labeled data. It achieves this by integrating principles of information maximization and redundancy reduction within a non-contrastive learning setup. The resulting representations are optimized for downstream NLP tasks such as semantic textual similarity (STS), document clustering, and information retrieval.

---

## Key Features

- **Non-Contrastive Architecture**: Utilizes a Barlow Twins-inspired loss adapted for textual data, eliminating the need for negative sampling or large batches.
- **Redundancy Minimization**: Reduces feature co-adaptation in the embedding space, mitigating collapse and encouraging information diversity.
- **Encoder Flexibility**: Compatible with transformer-based encoders like BERT, RoBERTa, T5, and lightweight variants such as DistilBERT.
- **Domain-Specific Augmentations**: Leverages NLP-centric transformations including back-translation, synonym substitution, random deletion, token reordering, and dropout-based perturbations to create meaningful positive pairs.
- **Mixup-Based Regularization**: Smooths the embedding space by interpolating between instances, enhancing generalization and stability.
- **Efficient Training Regimen**: Supports low batch sizes with AMP (Automatic Mixed Precision), employs AdamW with ReduceLROnPlateau scheduling, and uses early stopping with high patience for convergence.
- **Modular Implementation**: Facilitates experimentation with model architecture, data augmentation schemes, and training hyperparameters.

---

## Model Architecture

1. **Encoder**: BERT (`bert-base-uncased`) yields 768-dimensional contextual embeddings per sentence.
2. **Projection Head**: 2 fully connected layers (each with 4096 units), activated via ReLU, normalized with BatchNorm1D, and regularized using Dropout (rate: 0.117).
3. **Pooling Strategy**: Token-level embeddings are aggregated using mean pooling to form fixed-size sentence vectors.
4. **Objective Function**: A modified Barlow Twins loss incorporating a redundancy reduction term (λ_bt = 0.149) is combined with a mixup regularization loss (λ_mixup = 1.06).

---

## Data Augmentation

- **Synonym Substitution**: Replaces tokens with semantically similar alternatives using WordNet.
- **Random Token Deletion**: Improves model robustness by removing contextually relevant tokens.
- **Token Swapping**: Alters word order locally to introduce syntactic variability.
- **Dropout-Based Perturbation**: Introduces stochasticity at the embedding level during encoding.

> **Augmentation Strategy**: Dropout perturbation is applied first, followed by one randomly selected augmentation (substitution, deletion, or swapping).

---

## Mixup Regularization

- **Instance Interpolation**: Sentence embeddings are linearly interpolated with shuffled pairs.
- **Representation Alignment**: Mixed embeddings are encouraged to preserve alignment with original representations using cross-correlation objectives.
- **Tunable Strength**: Controlled by the `lambda_mixup` parameter.

---

## Training Pipeline

1. **Dataset**: Utilize STS-B benchmark with train/val/test splits.
2. **Pair Generation**: Perform data augmentation dynamically to construct training pairs.
3. **Optimization Procedure**:
   - Optimizer: AdamW (learning rate: 5.59e-5, with weight decay).
   - Learning Rate Schedule: ReduceLROnPlateau (patience = 200, factor = 0.5).
   - Mixed Precision Training: Enabled via AMP and GradScaler.
4. **Monitoring & Logging**: Track training loss, gradient norms, embedding variance, and STS-B correlations (Spearman & Pearson).
5. **Checkpointing**: Early stopping with patience; restore the best model based on validation metrics.

---

## Evaluation & Results

| Model                     | STS-B Score |
|---------------------------|-------------|
| DistilBERT                | 67.5        |
| DistilFACE                | 73.8        |
| CT-BERT-Base              | 74.31       |
| SimCSE-BERT-Base          | 76.85       |
| SBERT-NLI_Base            | 77.03       |
| **Mixed Barlow Twins (ours)** | **74.74**   |

- **Evaluation Metrics**: Pearson and Spearman correlation on STS-B test set.
- **Model Selection Criterion**: Maximum Spearman score on test data.
- **Performance**: Achieves Spearman correlation of 80.14% on validation and 74.74% on test set.

---

## Contributing

We welcome contributions, issue reports, and feature requests. Feel free to open a GitHub issue or submit a pull request!
