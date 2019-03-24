# Experiments

**Research Question: If MNLMs fail to generalise, what model architecture would generalise for generation?**

## Pretrained Embeddings

We initialize the decoder of the Bottom Up and Top Down Attention Model with the 300 dimensional
[GloVe embeddings](https://nlp.stanford.edu/projects/glove/), which are pretrained on Wikipedia 2014 and Gigaword 5.
These embeddings are not further fine-tuned. 

### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.249  | 0.038         | 0.071         | 0.132         | 0.125         | N/A
5         | 0.276  | 0.079         | 0.135         | 0.263         | 0.25          | N/A

Performance on "brown dog" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.335  | 0.048         | 0.057         | 0.133         | 0             | N/A
5         | 0.364  | 0.338         | 0.425         | 0.667         | 1             | N/A

Performance on "big car" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.267  | 0.032         | 0.056         | 0.077         | 0             | N/A
5         | 0.316  | 0.065         | 0.096         | 0.154         | 0             | N/A

### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.303  | 0.014         | 0.011         | 0             | 0             | N/A
5         | 0.309  | 0.034         | 0.035         | 0             | 0             | N/A

Performance on "white car" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.252  | 0.179         | 0.269         | 0.395         | 0.5           | N/A
5         | 0.308  | 0.383         | 0.492         | 0.632         | 0.625         | N/A

