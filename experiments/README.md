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
adjective-noun pair, the sample is counted as true positive

### Show, Attend and Tell

#### Baseline Performance

BLEU-4: 0.278

#### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.256  | 0             | 0             | 0             | 0             | N/A  
5         | 0.299  | 0.024         | 0.011         | 0             | 0             | N/A
10        | 0.296  | 0.028         | 0.011         | 0.067         | 0             | N/A
50        | 0.300  | 0.076         | 0.08          | 0.2           | 0             | N/A
100       | 0.300  | 0.131         | 0.138         | 0.2           | 0             | N/A

Performance on "white car" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.271  | 0.136         | 0.239         | 0.364         | 0.5           | N/A
5         | 0.297  | 0.367         | 0.493         | 0.772         | 1             | N/A


#### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.251  | 0.041         | 0.045         | 0.045         | 0             | N/A
5         | 0.227  | 0.086         | 0.104         | 0.182         | 0             | N/A

Performance on "brown dog" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.259  | 0.017         | 0             | 0             | 0             | N/A
5         | 0.329  | 0.346         | 0.437         | 0.8           | 1             | N/A

The results suggest that the model does not generalise to unseen adjective-noun pairs. The recall for adjective-noun
pairs of a model that was trained on data excluding the pairs is in all cases significantly lower compared to the recall of models that
were trained without the pairs being excluded from the training set.

#### Case studies

To understand why the models are failing to describe the respective adjective-noun pairs, case studies are performed.
We analyse samples where the agreement among the target captions is very high (n >= 4).

The following files contain the visualized attention and the full decoding beam for every timestep for different
samples:
- [brown_dog_1.md](brown_dog_1.md)
- [brown_dog_2.md](brown_dog_2.md)
- [brown_dog_3.md](brown_dog_3.md)
- [white_car_1.md](white_car_1.md)
- [white_car_2.md](white_car_2.md)
