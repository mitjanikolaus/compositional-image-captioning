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

### Show, Attend and Tell

#### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.274  | 0.003         | 0             | 0             | 0             | N/A
5         | 0.270  | 0.007         | 0.011         | 0             | 0             | N/A
10        | 
50        | 
100       | 

Performance on "white car" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.246  | 0.208         | 0.262         | 0.316         | 0.375         | N/A
5         | 0.251  | 0.380         | 0.484         | 0.526         | 0.375         | N/A


#### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.224  | 0.022         | 0.024         | 0.026         | 0             | N/A
5         | 0.234  | 0.014         | 0.032         | 0.026         | 0             | N/A

Performance on "brown dog" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.273  | 0.014         |  0.011        | 0.067         | 0             | N/A                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
5         | 0.331  | 0.345         |  0.437        | 0.8           | 1             | N/A

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

The only case where the adjective-noun pair occurs in the generated sentences in [white_car_6.md](white_car_6.md). In
this example, the car is very prominent in the image and has only one color. In some other examples, the adjective
occurs early in the beam, but then disappears when it would be combined with the noun. In
[white_car_3.md](white_car_3.md) we can see an example where the object has multiple colors, and the model describes
only the color it has seen in relation to the object at training time.


### Bottom Up and Top Down Attention

#### Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.319  | 0             | 0             | 0             | 0             | N/A
5         | 0.321  | 0.003         | 0.011         | 0             | 0             | N/A 

Performance on "white car" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 0.262  | 0.227         | 0.302         |  0.395        | 0.5           | N/A 
5         | 0.264  | 0.244         | 0.302         |  0.395        | 0.5           | N/A 


#### Model trained with held out "white car"

Performance on held out test set ("white car"):

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         | 
5         | 

Performance on "brown dog" data:

Beam size | BLEU-4 | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------|--------| --------------| --------------| --------------| --------------| -------------
1         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
5         | 
