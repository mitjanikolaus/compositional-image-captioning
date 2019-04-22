# Pretrained Embeddings

We initialize the decoder of the Bottom Up and Top Down Attention Model with the 300 dimensional
[GloVe embeddings](https://nlp.stanford.edu/projects/glove/), which are pretrained on Wikipedia 2014 and Gigaword 5.
These embeddings are not further fine-tuned. 

## Model trained with heldout pairs
Beam size 1:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.03 | 0.06 | 0.0 | 0.0 | N/A | 
eat_man | 0.05 | 0.08 | 0.18 | 0.15 | 0.33 | 
green_bench | 0.05 | 0.0 | 0.0 | 0.33 | 0.0 | 
red_chair | 0.07 | 0.25 | 0.17 | N/A | N/A | 
ride_woman | 0.05 | 0.19 | 0.24 | 0.53 | 0.67 | 
sit_cat | 0.03 | 0.03 | 0.07 | 0.08 | 0.05 | 
small_plane | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 
white_car | 0.04 | 0.08 | 0.1 | 0.29 | N/A | 
wooden_table | 0.0 | 0.0 | 0.0 | 0.0 | N/A |

Beam size 5:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.07 | 0.06 | 0.07 | 0.0 | N/A | 
eat_man | 0.07 | 0.19 | 0.24 | 0.31 | 0.33 | 
green_bench | 0.05 | 0.0 | 0.2 | 0.33 | 1.0 | 
red_chair | 0.1 | 0.0 | 0.17 | N/A | N/A | 
ride_woman | 0.14 | 0.23 | 0.21 | 0.6 | 0.67 | 
sit_cat | 0.07 | 0.1 | 0.15 | 0.2 | 0.21 | 
small_plane | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 
white_car | 0.06 | 0.08 | 0.16 | 0.14 | N/A | 
wooden_table | 0.04 | 0.0 | 0.0 | 0.0 | N/A | 

## Beam occurrences

### Model trained with held out "white car"

white car:
![Beam Occurrences](beam_occurrences_butd_white_car_white_car_glove.png)

brown dog:
![Beam Occurrences](beam_occurrences_butd_white_car_brown_dog_glove.png)

big car:
![Beam Occurrences](beam_occurrences_butd_white_car_big_car_glove.png)

### Model trained with held out "brown dog"

brown dog:
![Beam Occurrences](beam_occurrences_butd_brown_dog_brown_dog_glove_2.png)

big car:
![Beam Occurrences](beam_occurrences_butd_brown_dog_big_car_glove.png)

### Model trained with held out "big car"

big car:
![Beam Occurrences](beam_occurrences_butd_big_car_big_car_glove.png)


# Pretrained embeddings for both input and output layer

## Model trained with held out "brown dog"

Performance on held out test set ("brown dog"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.01          | 0.023         | 0             | 0             | N/A
5         | 0.09          | 0.103         | 0.2           | 0             | N/A

## Model trained with held out "big car"

Performance on held out test set ("big car"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.006         | 0.008         | 0.038         | 0.5           | N/A
5         | 0             | 0             | 0             | 0             | N/A

## Model trained with held out "small dog"

Performance on held out test set ("small dog"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0             | 0             | 0             | 0             | 0
5         | 0.003         | 0             | 0             | 0             | 0
