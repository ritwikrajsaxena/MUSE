# MUSE: Multi-perspective Uncertainty Synthesis Ensemble

This repository provides the reference implementation and evaluation pipelines for MUSE, a post hoc failure prediction framework for frozen Vision Language Models. MUSE evaluates whether model predictions are likely to fail without modifying or retraining the underlying Vision Language Model.

The repository supports failure prediction for both zero shot image classification and cross modal image text retrieval.

## Repository Contents

```text
.
├── Classification failure_detection.ipynb
├── Caption Retrieval failure detection.ipynb
└── README.md
```

### Classification Failure Prediction

Evaluation pipeline for post hoc failure prediction in zero shot image classification.

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

Failure prediction performance is evaluated using AUROC, FPR at 95 percent TPR, and Cohen's d. 

## Cross Modal Retrieval Failure Prediction

Extends the failure prediction framework to cross modal image text retrieval.

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

## Reproducibility

The notebooks contain the complete experimental pipelines used to extract uncertainty signals, construct the MUSE feature representation, train the synthesis layer, and evaluate failure prediction performance.

All model backbones, datasets, evaluation settings, and random seeds should be specified in the corresponding notebook configuration before execution.


Datasets:
| Dataset                              | URL / DOI                                                                                                                        |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Flowers-102                          | [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)                         |
| EuroSAT                              | [https://doi.org/10.5281/zenodo.7711810](https://doi.org/10.5281/zenodo.7711810)                                                 |
| Caltech-101                          | [https://doi.org/10.1109/CVPR.2004.383](https://doi.org/10.1109/CVPR.2004.383)                                                   |
| CIFAR-10 / CIFAR-100                 | [https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) |
| UEC Food-100                         | [https://doi.org/10.1007/s00530-023-01088-9](https://doi.org/10.1007/s00530-023-01088-9)                                         |
| UCF-101                              | [https://doi.org/10.48550/arXiv.1212.0402](https://doi.org/10.48550/arXiv.1212.0402)                                             |
| DTD                                  | [https://www.robots.ox.ac.uk/~vgg/data/dtd/](https://www.robots.ox.ac.uk/~vgg/data/dtd/)                                         |
| Food-101                             | [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)       |
| Oxford-IIIT Pet                      | [https://doi.org/10.1109/CVPR.2012.6248092](https://doi.org/10.1109/CVPR.2012.6248092)                                           |
| Stanford Cars                        | [https://doi.org/10.1109/CVPR.2015.7299023](https://doi.org/10.1109/CVPR.2015.7299023)                                           |
| SUN397                               | [https://doi.org/10.1109/CVPR.2010.5539970](https://doi.org/10.1109/CVPR.2010.5539970)                                           |
| FGVC Aircraft                        | [https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)                     |
| UCI Multimodal Damage Identification | [https://doi.org/10.24432/C52P6P](https://doi.org/10.24432/C52P6P)                                                               |
| MHII                                 | https://doi.org/10.1016/j.ipm.2022.102977                                        |
| CrisisMMD                            | [https://arxiv.org/abs/1805.00713](https://arxiv.org/abs/1805.00713)                                                             |
| ASONAM17 Building Damage             | [https://doi.org/10.1145/3110025.3110109](https://doi.org/10.1145/3110025.3110109)                                               |

