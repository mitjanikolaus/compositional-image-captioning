# Bottom Up and Top Down Attention

BLEU-4 baseline (karpathy splits): 0.342

## Model trained with full coco training set
Beam size 1:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.02 | 0.01 | 0.07 | 0.0 | N/A
eat_man | 0.22 | 0.4 | 0.44 | 0.77 | 0.33
green_bench | 0.1 | 0.0 | 0.4 | 0.33 | 0.0
red_chair | 0.14 | 0.5 | 0.67 | N/A | N/A
ride_woman | 0.23 | 0.56 | 0.58 | 0.8 | 1.0
sit_cat | 0.49 | 0.67 | 0.75 | 0.74 | 0.88
small_plane | 0.29 | 0.66 | 0.65 | 0.8 | 0.5
white_car | 0.18 | 0.25 | 0.45 | 0.57 | N/A
wooden_table | 0.05 | 0.18 | 0.0 | 1.0 | N/A


Beam size 5:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.28 | 0.31 | 0.79 | 1.0 | N/A | 
eat_man | 0.22 | 0.42 | 0.56 | 0.77 | 0.33 | 
green_bench | 0.25 | 0.38 | 0.4 | 0.67 | 1.0 | 
red_chair | 0.24 | 0.5 | 0.67 | N/A | N/A | 
ride_woman | 0.34 | 0.61 | 0.76 | 0.87 | 1.0 | 
sit_cat | 0.6 | 0.78 | 0.85 | 0.86 | 0.91 | 
small_plane | 0.32 | 0.76 | 0.74 | 0.8 | 0.5 | 
white_car | 0.42 | 0.53 | 0.77 | 0.71 | N/A | 
wooden_table | 0.35 | 0.36 | 0.0 | 1.0 | N/A |

## Model trained with heldout pairs
Beam size 1:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.0 | 0.0 | 0.0 | 0.0 | N/A | 
eat_man | 0.05 | 0.08 | 0.15 | 0.31 | 0.0 | 
green_bench | 0.05 | 0.0 | 0.0 | 0.67 | 0.0 | 
red_chair | 0.03 | 0.0 | 0.33 | N/A | N/A | 
ride_woman | 0.04 | 0.08 | 0.15 | 0.4 | 0.67 | 
sit_cat | 0.0 | 0.01 | 0.01 | 0.03 | 0.04 | 
small_plane | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 
white_car | 0.04 | 0.05 | 0.13 | 0.0 | N/A | 
wooden_table | 0.0 | 0.0 | 0.0 | 0.0 | N/A |

Beam size 5:

Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)
-----|--------------| -------------| -------------| -------------| -----------
brown_dog | 0.01 | 0.03 | 0.0 | 0.0 | N/A | 
eat_man | 0.06 | 0.09 | 0.15 | 0.23 | 0.33 | 
green_bench | 0.1 | 0.0 | 0.0 | 0.67 | 0.0 | 
red_chair | 0.17 | 0.25 | 0.17 | N/A | N/A | 
ride_woman | 0.04 | 0.11 | 0.18 | 0.53 | 0.33 | 
sit_cat | 0.01 | 0.0 | 0.02 | 0.02 | 0.04 | 
small_plane | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 
white_car | 0.05 | 0.03 | 0.16 | 0.0 | N/A | 
wooden_table | 0.06 | 0.27 | 0.0 | 1.0 | N/A | 

#### Beam Occurrences

**Occurrences of "brown dog"**

Model trained with heldout brown dog:
![Beam Occurrences](beam_occurrences_butd_brown_dog_brown_dog.png)

Model trained with heldout white car:
![Beam Occurrences](beam_occurrences_butd_white_car_brown_dog.png)


**Occurrences of "white car"**

Model trained with heldout white car:
![Beam Occurrences](beam_occurrences_butd_white_car_white_car.png)

Model trained with heldout brown dog:
![Beam Occurrences](beam_occurrences_butd_brown_dog_white_car.png)


