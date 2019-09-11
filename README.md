# compositional-image-captioning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This repository contains code for evaluating compositional generalization in image captioning models.
It additionally contains code for two state-of-the-art image captioning models and a joint model for caption generation and image-sentence ranking, called BUTR.
All models are implemented in PyTorch.


It accompanies the following CoNLL 2019 paper:

[Compositional Generalization in Image Captioning](https://arxiv.org/abs/1909.04402)    
Mitja Nikolaus, Mostafa Abdou, Matthew Lamm, Rahul Aralikatte and Desmond Elliott


## Models

### Show, Attend and Tell

An implementation of the
[Show, Attend and Tell](https://arxiv.org/abs/1502.03044) model
(adapted from a
[PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)).

### Bottom-Up and Top-Down Attention

An implementation of the decoder of the
[Bottom-Up and Top-Down Attention](https://arxiv.org/abs/1707.07998) model.
Encoded features from the bottom-up attention model for the COCO dataset
can be found on the
[project's GitHub page](https://github.com/peteanderson80/bottom-up-attention).

### Bottom-Up and Top-Down attention with Re-ranking (BUTR)

An implementation of the joint model for caption generation and
image-sentence ranking based on the
[Bottom-Up and Top-Down Attention](https://arxiv.org/abs/1707.07998)
and [VSE++](https://arxiv.org/abs/1707.05612) models.

## Compositional Generalization Evaluation

A model has to be trained and evaluated using the four different
[dataset splits](data/dataset_splits). Afterwards, the resulting
evaluation json files should be merged into a single json containing
the results for all 24 held out concept pairs. The average recall@5
and some other statistics can be visualized using
[plot_recall_results.py](plot_recall_results.py)


