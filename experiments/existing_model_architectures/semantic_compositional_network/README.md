# Experiments

## 1. Generalisation Capabilities

**Research Question: How well do Multimodal Neural Language Models (MNLMs) generalise to unseen
Adjective-Noun (Adj-N) pairs?**

All images of the COCO dataset were annotated with information about how often a specific adjective-noun pair
combination occurs in the corresponding captions. 

Subsets of the COCO training set were created by removing all samples where a specific adjective-noun pair occurs at
least once in the caption. For each of the training sets, a model was trained. 

For the evaluation, data from the COCO validation set was annotated in the same way. This time, only samples that where
the adjective-noun pair occurs are added to the test set.

Notes on how to interpret the results:
- "Recall (n>=1)" stands for the recall of the respective adjective noun pair,
where n of the target captions contain the adjective noun pair.
- For a beam size of n, the top n sentences are produced. If at least one of the sentences contains the target
adjective-noun pair, the sample is counted as true positive (i.e. we are calculating the recall@n)

### Semantic Compositional Network

BLEU-4 baseline (karpathy splits): 0.327

#### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.003         | 0             | 0             | 0             | N/A
5         | 0.007         | 0             | 0             | 0             | N/A

Performance on "white car" data:

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.187         | 0.261         | 0.342         | 0.5           | N/A
5         | 0.323         | 0.444         | 0.526         | 0.625         | N/A

#### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.012         | 0.016         | 0             | 0             | N/A
5         | 0.026         | 0.032         | 0.079         | 0.125         | N/A

Performance on "brown dog" data:

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.007         | 0             | 0             | 0             | N/A
5         | 0.193         | 0.299         | 0.467         | 0             | N/A

