# Bottom-Up Ranking Generation

We design a network for a multi-task scenario: The first objective is to learn visual-semantic embeddings for
cross-modal retrieval. The second objective is to generate captions for a given image. As image features, the
pre-trained features from the Bottom-Up-Top-Down model are used.

## Training with ranking objective

Recall@5 for the target pair when ranking the whole coco validation set (40504 samples):

Pair | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
brown_dog | 0.37 | 0.43 | 0.57 | 1.0 | N/A | 
eat_man | 0.26 | 0.58 | 0.68 | 0.77 | 0.33 | 
green_bench | 0.5 | 0.5 | 0.4 | 0.67 | 1.0 | 
red_chair | 0.24 | 0.5 | 0.67 | N/A | N/A | 
ride_woman | 0.36 | 0.67 | 0.79 | 0.73 | 1.0 | 
sit_cat | 0.62 | 0.79 | 0.88 | 0.91 | 0.91 | 
small_plane | 0.39 | 0.76 | 0.81 | 0.8 | 0.5 | 
white_car | 0.3 | 0.51 | 0.55 | 1.0 | N/A | 
wooden_table | 0.12 | 0.45 | 0.0 | 1.0 | N/A |

## Training with joint objective
Generation performance:

Beam size 1:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.01 | 0.03 | 0.0 | 0.0 | N/A | 
eat_man | 0.03 | 0.09 | 0.12 | 0.23 | 0.33 | 
green_bench | 0.05 | 0.0 | 0.2 | 0.33 | 0.0 | 
red_chair | 0.03 | 0.25 | 0.17 | N/A | N/A | 
ride_woman | 0.05 | 0.11 | 0.15 | 0.4 | 0.33 | 
sit_cat | 0.01 | 0.03 | 0.05 | 0.08 | 0.09 | 
small_plane | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 
white_car | 0.01 | 0.0 | 0.06 | 0.0 | N/A | 
wooden_table | 0.0 | 0.0 | 0.0 | 0.0 | N/A |

Beam size 5:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.05 | 0.04 | 0.14 | 0.0 | N/A | 
eat_man | 0.05 | 0.09 | 0.15 | 0.23 | 0.0 | 
green_bench | 0.15 | 0.0 | 0.0 | 0.33 | 0.0 | 
red_chair | 0.14 | 0.5 | 0.33 | N/A | N/A | 
ride_woman | 0.11 | 0.22 | 0.18 | 0.4 | 0.33 | 
sit_cat | 0.03 | 0.06 | 0.09 | 0.14 | 0.16 | 
small_plane | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 
white_car | 0.05 | 0.08 | 0.23 | 0.14 | N/A | 
wooden_table | 0.11 | 0.18 | 0.0 | 0.0 | N/A |

## First train on ranking objective, then generation

### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.041         | 0.048         | 0.079         | 0             | N/A
5         | 0.103         | 0.135         | 0.237         | 0             | N/A

### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.01          | 0.02          | 0             | 0             | N/A
5         | 0.031         | 0.057         | 0.133         | 0             | N/A


## First train on ranking objective, then generation with frozen embedding weights

### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.038         | 0.063         | 0.105         | 0.125         | N/A
5         | 0.093         | 0.159         | 0.289         | 0.25          | N/A

### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.014         | 0.034         | 0.067         | 0             | N/A
5         | 0.03          | 0.046         | 0.067         | 0            | N/A

