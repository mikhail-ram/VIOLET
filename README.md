# VIOLET: Vectorized Invariance Optimization for Language Embeddings using Twins

## Project Overview
This project fine-tunes DistilBERT using a non-contrastive learning approach inspired by Barlow Twins, focusing on learning robust sentence embeddings. The model is trained on augmented views of the same sentence to maximize information retention while ensuring invariance to perturbations.

### Primary Goals:
- Generate high-quality sentence embeddings using a self-supervised approach.
- Leverage contrastive learning with only positive pairs—no in-batch negatives.
- Use text augmentations to create diverse training pairs and improve generalization.
- Employ MixUp regularization to prevent overfitting and encourage smoother representations.
- Evaluate on STS Benchmark (STS-B) development and test sets using Spearman correlation as the primary metric.

## Model and Training Methodology

### 1. Base Model: DistilBERT
DistilBERT, a lighter version of BERT, is fine-tuned to output contextualized token embeddings, which are pooled to obtain sentence embeddings.

### 2. Data Augmentation for Contrastive Learning
Text augmentations create two different views of the same sentence to serve as positive pairs:
- **Synonym Replacement**: Replaces words with synonyms to add lexical diversity.
- **Word Swap**: Randomly swaps words to introduce syntactic variation.
- **Random Deletion**: Removes words, forcing the model to generalize.
- **Dropout**: Applied across all augmentations to increase robustness.

These augmentations encourage the model to focus on semantic meaning rather than superficial token variations.

### 3. Contrastive Loss: Barlow Twins Adaptation
Instead of using traditional contrastive loss (e.g., NT-Xent), this project employs Barlow Twins, which maximizes similarity between embeddings of augmented pairs while ensuring that redundant features are suppressed.

#### Loss Function: Mixed Barlow Twins
Barlow Twins loss is computed as:
$$L = \sum (I - C)^2$$
where $C$ is the cross-correlation matrix of embeddings from two augmentations.

**Key components:**
- **Invariance Term**: Ensures that augmented versions of the same sentence are mapped to similar embeddings.
- **Redundancy Reduction Term**: Prevents collapsed representations by decorrelating feature dimensions.

The "mixed" Barlow Twins variant introduces MixUp regularization, which interpolates embeddings between different augmentations. This reduces overfitting by ensuring smooth transitions in the embedding space.

### 4. MixUp Regularization
#### Why?
Without MixUp, initial experiments showed that the model overfitted quickly, performing well on validation but failing on the test set.

#### How?
Given two embeddings $z_1, z_2$ from augmentations of the same sentence, the mixed embedding is computed as:
$$z_{mix} = \lambda z_1 + (1 - \lambda) z_2$$
where $\lambda$ is drawn from a Beta distribution.

The loss is computed with respect to both original embeddings, forcing the model to generalize better. This prevents memorization of augmentations and instead encourages robust sentence representations.

### 5. Training Procedure
- Positive pairs are generated using augmentations.
- Barlow Twins loss is applied to enforce invariance.
- MixUp regularization prevents overfitting.
- **AdamW** optimizer is used for efficient gradient updates.
- Learning rate scheduling with warmup ensures smooth convergence.
- Early stopping based on validation Spearman correlation prevents over-training.

## Evaluation: STS Benchmark (STS-B)
The model is evaluated using the STS-B dataset, which measures the semantic similarity of sentence pairs.

### Evaluation Process:
1. Encode the STS sentence pairs using the fine-tuned DistilBERT.
2. Compute cosine similarity between sentence embeddings.
3. Compare results against human-labeled similarity scores using **Spearman correlation**.

Spearman correlation captures ranking consistency, making it a strong metric for evaluating sentence embeddings.

### Why STS-B?
- A standard benchmark for sentence similarity.
- Used in SimCSE, Sentence-BERT, and other state-of-the-art models.
- Ensures that embeddings generalize beyond training pairs.
