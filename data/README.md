# captioning-models data

## Numbers of matching adjective-noun pairs in the COCO dataset

Target | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
white car | 839.0 | 274.0 | 101.0 | 31.0 | 5.0
car | 7410.0 | 5576.0 | 4674.0 | 3950.0 | 2826.0
white | 16709.0 | 5476.0 | 1759.0 | 479.0 | 85.0


Captions are counted as matches also when one of the synonyms for the respective adjective or nouns is present (see the
respective JSON files for a list of synonyms considered). For adjective-noun pair matches, we only consider captions
where the adjective is also describing the target noun (using the
[stanfordnlp dependency parser](https://github.com/stanfordnlp/stanfordnlp)).

'N' denotes the agreement among the different captions, e.g. 
'N=3' means that 3 of the 5 image captions contain the target.
