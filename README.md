# MUSE: Multi-perspective Uncertainty Synthesis Ensemble

This repository provides the reference implementation and evaluation pipelines for MUSE, a post hoc failure prediction framework for frozen Vision Language Models. MUSE evaluates whether model predictions are likely to fail without modifying or retraining the underlying Vision Language Model.

The repository supports failure prediction for both zero shot image classification and cross modal image text retrieval.

## Repository Contents

```text
.
├── classification_failure.ipynb
├── caption_retrieval_failure_pipeline.ipynb
├── requirements.txt
└── README.md
```

### Classification Failure Prediction

`classification_failure.ipynb` contains the complete evaluation pipeline for post hoc failure prediction in zero shot image classification.

The pipeline evaluates frozen Vision Language Models on 20 general purpose and disaster specific classification benchmarks using the following backbones:

| Model    | Backbone | Implementation                   |
| -------- | -------- | -------------------------------- |
| CLIP     | ViT B/32 | `openai/clip-vit-base-patch32`   |
| SigLIP   | ViT B/16 | `google/siglip-base-patch16-224` |
| OpenCLIP | ViT B/32 | `laion2b_s34b_b79k`              |

The classification pipeline consists of the following stages.

1. Zero shot inference

The Vision Language Model computes image text similarities and produces a class prediction. Each prediction is assigned a failure label according to whether the predicted class matches the ground truth.

2. Failure signal extraction

The pipeline extracts complementary uncertainty signals from several post hoc failure prediction methods:

* ATLAS
* SOARER
* ICS
* AugLikelihoodQDA
* SWNC


3. MUSE synthesis

MUSE combines the individual uncertainty signals into a low dimensional feature representation and learns a regularized meta classifier using a calibration split.

4. Evaluation

Failure prediction performance is evaluated using AUROC, FPR at 95 percent TPR, and AURC. The evaluation supports multiple random seeds and pairwise statistical significance testing using the Wilcoxon signed rank test.

## Cross Modal Retrieval Failure Prediction

`caption_retrieval_failure_pipeline.ipynb` extends the failure prediction framework to cross modal image text retrieval.

The pipeline evaluates both text to image and image to text retrieval and treats retrieval misses as prediction failures.

The retrieval pipeline consists of the following stages.

1. Multimodal embedding

Images and captions are encoded using the frozen vision and text encoders of the selected Vision Language Model. Embeddings are normalized before computing cross modal similarities.

2. Retrieval evaluation

The pipeline constructs the image text similarity matrix and determines the retrieval rank of the correct target. Retrieval success or failure is defined according to the selected retrieval criterion, such as Rank 1 or Recall at K.

3. Retrieval uncertainty signals

The pipeline extracts retrieval specific signals, including:

* Bidirectional cross modal alignment margins
* Nearest neighbor ambiguity
* Visual textual consistency under semantic perturbations

4. MUSE synthesis

The retrieval uncertainty signals are combined into a unified failure risk score.

5. Evaluation

The resulting scores are evaluated using failure detection metrics and selective retrieval analysis to measure how effectively high risk queries can be identified for filtering or human review.

## Environment

The implementation requires Python 3.9 or later and is designed to run with PyTorch 2.0 or later. CUDA is recommended for Vision Language Model inference.

The primary dependencies include:

```text
torch
torchvision
open_clip_torch
transformers
scikit-learn
scipy
numpy
pandas
tqdm
matplotlib
```

## Installation

A CUDA enabled installation can be configured with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch transformers scikit-learn scipy numpy pandas tqdm matplotlib
```

Alternatively, the complete environment can be installed from `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Running the Classification Pipeline

Open:

```text
classification_failure.ipynb
```

Set the desired dataset through `DATASET_NAME`. For example:

```python
DATASET_NAME = "oxford_pets"
```

Other supported configurations include datasets such as:

```python
DATASET_NAME = "cifar100"
DATASET_NAME = "crisismmd_dmg"
```

Run the notebook to perform zero shot inference, extract failure prediction signals, train the MUSE synthesis layer, and generate the evaluation results.

## Running the Retrieval Pipeline

Open:

```text
caption_retrieval_failure_pipeline.ipynb
```

Select the retrieval benchmark and retrieval direction:

```python
direction = "image_to_text"
```

or:

```python
direction = "text_to_image"
```

Run the notebook to generate the cross modal similarity matrix, compute retrieval ranks, extract failure prediction signals, synthesize the MUSE risk score, and evaluate failure detection performance.

## Evaluation

For classification, the primary evaluation metrics are:

* AUROC
* FPR at 95 percent TPR
* AURC

For retrieval, the evaluation includes:

* AUROC
* Retrieval failure detection performance
* Selective retrieval and risk coverage analysis

Multiple random seeds can be used to measure variability across runs. Statistical comparisons between failure prediction methods are performed using the Wilcoxon signed rank test where applicable.

## Reproducibility

The notebooks contain the complete experimental pipelines used to extract uncertainty signals, construct the MUSE feature representation, train the synthesis layer, and evaluate failure prediction performance.

All model backbones, datasets, evaluation settings, and random seeds should be specified in the corresponding notebook configuration before execution.
