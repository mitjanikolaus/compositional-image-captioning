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

#### Model trained with held out "big car"

Performance on held out test set ("white car"):

Beam size | Recall (n>=1) | Recall (n>=2) | Recall (n>=3) | Recall (n>=4) | Recall (n>=5)
----------| --------------| --------------| --------------| --------------| -------------
1         | 0.006         | 0.008         | 0             | 0             | N/A
5         | 0.002         | 0.008         | 0.038         | 0             | N/A


#### Qualitative Analysis for model trained with heldout "brown dog"

For n>=3, the results look as follows:

Average probability of tag 'dog': 0.98

Average probability of tag 'brown': 0.74

Recall@5 for 'brown' in the generated captions: 0
```
187199
a dog laying on top of a bed
a large dog laying on top of a bed
a dog laying on a bed next to a book
a dog laying on top of a bed next to a book
a dog laying on top of a bed with a blanket

537955
a dog that is standing in a kitchen
a dog that is standing in a room
a dog standing in a room with a wooden floor
a dog is standing in a room with a door
a dog is standing in a room with a wooden floor

214984
a dog with a frisbee in its mouth
a dog is holding a frisbee in its mouth
a dog running with a frisbee in its mouth
a dog holding a frisbee in its mouth
a dog with a frisbee in its mouth in a field

283210
a dog with a frisbee in its mouth
two dogs are playing with a frisbee in the grass
a dog laying on the grass with a frisbee in its mouth
a dog laying on the grass with a frisbee
a dog laying on top of a green field with a frisbee in its mouth

114891
a dog sitting on the floor next to a teddy bear
a dog sitting on the floor next to a stuffed animal
a dog sitting on a couch next to a teddy bear
a dog sitting on the floor next to a dog
a dog sitting on a couch next to a stuffed animal

520301
a dog looking out the window of a car
a dog is looking out the window of a car
a dog sitting in the back of a car
a dog that is looking out the window
a dog sitting in the back of a car window

54643
a dog is laying on a couch
a dog laying on a couch next to a cat
a dog laying on a couch next to a dog
a dog laying on a couch next to a stuffed animal
a dog laying on a couch next to a stuffed bear

29393
a dog standing in front of a house
a dog and a dog standing in front of a house
a dog standing on a sidewalk next to a house
a dog standing next to a dog on a sidewalk
a dog standing in front of a house with a dog

443139
a dog sitting in the back of a car
a dog sitting in the back seat of a car
a dog is sitting in the back of a car
a dog laying in the back of a car
a dog sitting in the back of a car with a dog in the back seat

251572
a dog is laying down on a couch
a dog laying on the floor next to a woman
a dog laying on the floor next to a person
a dog laying on a couch next to a person
a dog is laying on the floor next to a woman

89109
a dog that is laying on a bed
a dog that is laying down on a bed
a dog that is laying down on a blanket
a dog laying on a bed with its head on its head
a dog laying on a bed with its head on a pillow

307423
a black and white cat sitting on a bench
a black and white dog sitting on a bench
a cat sitting on the ground next to a dog
a cat sitting on the ground next to a cat
a cat sitting on a bench next to a cat

423036
a dog laying on the ground next to a window
a dog laying on the ground next to a wall
a dog laying on the ground in front of a window
a dog laying on the ground in front of a building
a large dog laying on the ground next to a window

42856
a dog that is sitting on a couch
a dog is sitting on a couch with a remote
a dog sitting on a couch with a cat on it
a dog laying on a couch with a cat on it
a dog is sitting on a couch with a cat

476767
a large dog laying on the floor in front of a mirror
a large dog laying on the floor next to a cat
a large dog laying on the floor next to a table
a large dog laying on the floor next to a glass
a large dog laying on the floor next to a glass of wine
```

**Forcing probability of 'dog' and 'brown' to be 1:**

Recall@5 for 'brown' in the generated captions: 0
```
187199
a large dog laying on top of a bed
a dog laying on top of a bed
a dog laying on its back on a bed
a dog laying on top of a pillow on a bed
a dog laying on top of a bed with a blanket

537955
a dog that is standing in a kitchen
a dog that is standing in a room
a dog standing next to a wooden floor in a kitchen
a dog standing next to a wooden floor in a room
a dog is standing in a room with a door

214984
a dog with a frisbee in its mouth
a dog is holding a frisbee in its mouth
a dog holding a frisbee in its mouth
a dog is playing with a frisbee in its mouth
a dog with a frisbee in its mouth in a field

283210
a dog with a frisbee in its mouth
a dog laying on the grass with a frisbee in its mouth
a dog laying on the ground with a frisbee in its mouth
a dog laying on top of a grass covered field
a dog laying on the grass with a frisbee

114891
a dog sitting on the floor next to a teddy bear
a dog sitting on the floor next to a stuffed animal
a dog sitting on a couch next to a teddy bear
a dog sitting on the floor next to a dog
a dog sitting on a couch next to a stuffed animal

520301
a dog looking out the window of a car
a dog is looking out the window of a car
a dog sitting in the back of a car
a dog that is looking out the window
a dog sitting in the back of a car window

54643
a dog is laying on a couch
a dog laying on a couch next to a cat
a dog laying on a couch next to a teddy bear
a dog laying on a couch next to a stuffed animal
a dog laying on a couch next to a stuffed bear

29393
a dog and a dog standing in front of a house
a dog and a dog standing on a sidewalk
a dog standing on a sidewalk next to a house
a dog standing next to a dog on a sidewalk
a dog and a dog standing on the side of a road

443139
a dog sitting in the back of a car
a dog is sitting in the back of a car
a dog sitting in the back seat of a car
a dog laying in the back of a car
a dog sitting in the back of a car with a dog in the back seat

251572
a dog is laying down on a couch
a dog laying on the floor next to a woman
a dog laying on the floor next to a person
a dog laying on a couch next to a person
a dog is laying on the floor next to a woman

89109
a dog that is laying down on a bed
a dog that is laying down in a bed
a dog that is laying down on a blanket
a dog laying on a bed with a blanket
a dog laying on a bed with its head on its head

307423
a couple of dogs sitting next to each other
a couple of dogs sitting next to each other on a bench
a couple of dogs sitting on top of a bench
a couple of dogs sitting next to each other on the ground
a couple of dogs sitting next to each other on a couch

423036
a close up of a dog laying on the ground
a close up of a dog laying on the floor
a close up of a dog laying on a floor
a close up of a dog laying down on the ground
a close up of a dog laying on the ground next to a wall

42856
a dog that is sitting on a couch
a dog is sitting on a couch with a remote
a dog sitting on a couch with a cat on it
a dog laying on a couch with a cat on it
a dog is sitting on a couch with a cat

476767
a large dog laying on the floor in front of a mirror
a large dog laying on the floor next to a cat
a large dog laying on the floor next to a table
a large dog laying on the floor next to a glass
a large dog laying on the floor next to a glass of wine
```

**Forcing probability of 'dog' and 'brown' to be 1, all others 0:**

Recall@5 for 'brown' in the generated captions: 0.466
```
187199
a dog a dog
a **brown** dog a dog
a small dog a dog
a dog a dog a
a **brown** dog a dog in a

537955
a **brown** dog in a
a **brown** dog in in a
a **brown** dog in a a
a **brown** dog in a in a
a **brown** dog in in a a

214984
a **brown** dog in a
a **brown** dog in a a
a **brown** dog in a dog
a **brown** dog in a in a
a **brown** dog in a in a a

283210
a small dog a dog
a **brown** dog a dog
a dog a dog in a
a small dog a dog in a
a small dog a dog in a a

114891
a small dog a dog
a **brown** dog a dog
a dog a dog in a
a small dog a dog in a
a small dog a dog in a a

520301
a small dog a dog
a dog a a dog
a dog a dog in a
a small dog a dog in a
a small dog a dog in a a

54643
a dog a dog
a small dog a dog
a **brown** dog with a
a **brown** dog with a dog
a **brown** dog with a a

29393
a small dog a dog
a dog is in a
a dog a dog in a
a dog is in a a
a dog a dog in a a

443139
a small dog a dog
a a dog a dog
a dog is a dog
a small dog a dog in
a small dog a dog in a

251572
a dog a dog
a small dog a dog
a dog a dog in
a dog a dog in a
a dog a dog in a a

89109
a dog in a
a dog with a dog
a dog in a dog
a dog in a a
a dog in a a dog

307423
a a dog a dog
a a dog is a
a a dog a dog in
a a dog a dog in a
a a dog a dog in a a

423036
a a dog in a
a a dog a dog
a a dog with a
a a dog in a a
a a dog in a a dog

42856
a dog a dog
a dog is a dog
a dog a dog in
a dog a dog in a
a dog a dog in a a

476767
a **brown** dog a dog
a a dog a dog
a dog a dog in a
a **brown** dog a dog in a
a **brown** dog a dog in a a
```

**Forcing probability of 'dog' and 'brown' to be 1, all others reduced by 0.5:**

Recall@5 for 'brown' in the generated captions: 0.267
```
187199
a dog laying on a bed
a large dog laying on a bed
a small dog laying on a bed
a dog is laying on a bed
a dog laying on a bed with a dog

537955
a dog is standing in a room
a dog is standing in a kitchen
a dog with a dog in a room
a dog standing in a room with a dog
a dog is standing in a room with a dog

214984
a dog with a frisbee in its mouth
a dog holding a frisbee in its mouth
a dog running with a frisbee in its mouth
a dog is holding a frisbee in its mouth
a dog with a frisbee in its mouth in the air

283210
a dog with a dog with a frisbee
a dog with a dog in a frisbee
a dog with a dog in a field
a dog with a frisbee in a field
a dog with a dog with a frisbee in its mouth

114891
a dog with a dog in a dog
a dog with a dog in a toy
a small dog with a dog in its mouth
a small dog with a dog in a dog
a small dog with a dog in a toy

520301
a dog with a dog in a car
a dog with a dog in a dog
a dog with a dog in the back
a dog with a dog with a dog
a dog with a dog in the back of a car

54643
a dog with a dog
a small dog with a dog
a **brown** dog with a dog
a small dog with a a dog
a small dog with a dog in a

29393
a dog standing in the back of a house
a dog standing in the back of a fence
a dog standing in the back of a dog
a dog is standing in the front of a house
a dog is standing in the front of a fence

443139
a dog with a dog in a car
a dog is in the back of a car
a dog with a dog on a car
a dog is in the back of a vehicle
a dog with a dog in the back of a car

251572
a dog is laying in a bed
a dog is laying on a bed
a dog is laying on the back of a
a dog is laying on the back of a horse
a dog is laying on the back of a person

89109
a dog is looking at a dog
a dog with a dog in its head
a dog with a dog in a car
a dog with a dog in a dog
a dog with a dog in a toy

307423
a couple of **brown** dogs and a dog
a couple of **brown** dogs and a cat
a couple of **brown** dogs and a **brown** dog
a couple of **brown** dogs and a dog in a bed
a couple of **brown** dogs and a dog on a floor

423036
a **brown** dog with a dog
a **brown** dog with a dog in a
a small dog with a dog in a
a **brown** dog with a dog in a a
a small dog with a dog in a a

42856
a dog is sitting on a couch
a dog is sitting on a bed
a dog is sitting on a **brown** couch
a dog sitting on a couch with a dog
a dog is sitting on a couch with a dog

476767
a dog is laying on the floor
a dog is laying on a couch
a dog with a dog laying on the floor
a dog with a dog laying on the ground
a dog with a dog laying on a floor
```