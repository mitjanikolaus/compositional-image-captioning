# captioning-models data

## Numbers of matching adjective-noun pairs in the COCO dataset

Pair | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| ---------------|-----------------|----------------|--------------
white car | 839.0 | 274.0 | 101.0 | 31.0 | 5.0
big car | 1041.0 | 239.0 | 60.0 | 15.0 | 2.0
brown dog | 614.0 | 164.0 | 32.0 | 3.0 | 0.0
small cat | 259.0 | 57.0 | 15.0 | 2.0 | 0.0 

## Noun occurrences in the COCO dataset

Noun | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
car | 7410.0 | 5576.0 | 4674.0 | 3950.0 | 2826.0
dog | 3625.0 | 3281.0 | 3011.0 | 2645.0 | 2062.0
cat | 2859.0 | 2723.0 | 2650.0 | 2562.0 | 2223.0

## Adjective occurrences in the COCO dataset

Adjective | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
white | 16709.0 | 5476.0 | 1759.0 | 479.0 | 85.0
brown | 4335.0 | 1029.0 | 281.0 | 64.0 | 11.0
big | 18966.0 | 6007.0 | 1994.0 | 611.0 | 128.0
small | 13783.0 | 4225.0 | 1423.0 | 408.0 | 63.0 


Captions are counted as matches also when one of the synonyms for the respective adjective or nouns is present (see the
respective JSON files for a list of synonyms considered). For adjective-noun pair matches, we only consider captions
where the adjective is also describing the target noun (using the
[stanfordnlp dependency parser](https://github.com/stanfordnlp/stanfordnlp)).

'N' denotes the agreement among the different captions, e.g. 
'N=3' means that 3 of the 5 image captions contain the target.
