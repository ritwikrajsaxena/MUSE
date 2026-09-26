MUSE: Failure Prediction Pipelines
This repository provides the reference implementation and evaluation pipelines for MUSE (Multi-perspective Uncertainty Synthesis Ensemble), evaluating failure prediction for frozen Vision-Language Models (VLMs) across classification and cross-modal retrieval tasks.

📁 Repository Structure
text

.
├── classification_failure.ipynb            # End-to-end classification failure detection pipeline
├── caption_retrieval_failure_pipeline.ipynb # Cross-modal retrieval failure detection pipeline
├── requirements.txt                         # Environment dependencies
└── README.md                                # This document
⚙️ Environment Setup
1. Requirements
Python 
≥
≥ 3.9
PyTorch 
≥
≥ 2.0 (CUDA recommended)
open_clip_torch, transformers, scikit-learn, scipy, numpy, pandas, tqdm
2. Quick Install
Bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch transformers scikit-learn scipy numpy pandas tqdm matplotlib
📓 Notebook 1: classification_failure.ipynb
Overview
Evaluates post-hoc failure detection on zero-shot image classification across 20 general and disaster-specific benchmarks using frozen VLMs (CLIP ViT-B/32, SigLIP ViT-B/16, and OpenCLIP ViT-B/32).

Pipeline Stages
Zero-Shot Inference: Computes text-image cosine similarities and determines ground-truth success/failure labels (
y
fail
=
I
(
y
^
≠
y
)
y 
fail
​
 =I( 
y
^
​
 

=y)).
Individual Detector Feature Extraction:
ATLAS: Evaluates cross-modal alignment collapse in the joint embedding manifold.
SOARER: Quantifies semantic competition across candidate class prompts.
ICS: Measures intra-class representation collapse and class-conditional variance.
AugLikelihoodQDA: Fits regularized density estimators over augmented embedding space.
SWNC: Computes sample-wise neighborhood consistency across visual perturbations.
Baseline Comparison: Extracts uncertainty signals for MSP, MCM, LVU/TCP, ViLU, and TrustVLM.
MUSE Meta-Ensemble Training: Fits a regularized meta-classifier on a low-dimensional feature space (
s
∈
R
5
s∈R 
5
 ) generated from the calibration split.
Evaluation & Metrics: Reports AUROC, FPR@95, and AURC across multiple random seeds, including pairwise statistical significance tests (Wilcoxon signed-rank).
📓 Notebook 2: caption_retrieval_failure_pipeline.ipynb
Overview
Extends MUSE to cross-modal caption and image retrieval (Text-to-Image and Image-to-Text), framing retrieval misses (Rank-1 / Recall@
K
K failures) as prediction errors to be detected without querying ground-truth targets.

Pipeline Stages
Multimodal Indexing & Embedding: Encodes image and caption galleries into normalized metric spaces using frozen VLM vision and text encoders.
Retrieval Simulation: Computes cross-modal similarity matrices and logs retrieval ranks (success: top-1 or top-
K
K hit; failure: retrieval miss).
Retrieval-Adapted Uncertainty Extraction:
Computes bi-directional cross-modal alignment margins.
Quantifies nearest-neighbor ambiguity in the retrieval gallery.
Measures visual-textual consistency under semantic perturbations.
Meta-Ensemble Synthesis: Synthesizes retrieval-specific uncertainty features into a unified risk score.
Failure Detection Benchmarking: Computes AUROC and selective-retrieval curves evaluating how effectively high-risk queries can be filtered or routed to human review.
📊 Summary of Evaluated Backbones
Model Family	Backbone	Source / Implementation
CLIP	ViT-B/32	openai/clip-vit-base-patch32
SigLIP	ViT-B/16	google/siglip-base-patch16-224
OpenCLIP	ViT-B/32	laion2b_s34b_b79k
🚀 Quick Execution Guide
Classification:

Open classification_failure.ipynb.
Set DATASET_NAME (e.g., 'oxford_pets', 'cifar100', 'crisismmd_dmg').
Run all cells to extract detector scores, train the MUSE fusion layer, and output the AUROC/FPR@95 summary table.
Retrieval:

Open caption_retrieval_failure_pipeline.ipynb.
Select retrieval benchmark and direction ('image_to_text' or 'text_to_image').
Execute to compute retrieval rankings, evaluate failure detector signals, and generate risk-coverage trade-off curves.
