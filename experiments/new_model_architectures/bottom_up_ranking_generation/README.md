# Bottom-Up Ranking Generation

We design a network for a multi-task scenario: The first objective is to learn visual-semantic embeddings for
cross-modal retrieval. The second objective is to generate captions for a given image. As image features, the
pre-trained features from the Bottom-Up-Top-Down model are used.

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

