# captioning-models

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This repository contains models for image captioning. All models are implemented in PyTorch.

## Models

### Show, Attend and Tell

An implementation of the [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) model
(adapted from a
[PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)).

### Bottom-Up and Top-Down Attention

An implementation of the decoder of the [Bottom-Up and Top-Down Attention](https://arxiv.org/abs/1707.07998) model.
Encoded features from the bottom-up attention model for the COCO dataset can be found on the
[project's GitHub page](https://github.com/peteanderson80/bottom-up-attention).

## Experiments

Descriptions and results of performed experiments can be found in the [experiments directory](experiments/). Some
overview of the data used to run the experiments can be found in the [data directory](data/).