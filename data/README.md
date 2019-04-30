# captioning-models data

Captions are counted as matches also when one of the synonyms for the respective adjective or nouns is present (see the
respective JSON files for a list of synonyms considered). For adjective-noun pair matches, we only consider captions
where the adjective is also describing the target noun (using the
[stanfordnlp dependency parser](https://github.com/stanfordnlp/stanfordnlp)).

'N' denotes the agreement among the different captions, e.g. 
'N=3' means that 3 of the 5 image captions contain the target.

## Numbers of matching adjective-noun or verb-noun pairs

### COCO training dataset

Pair | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5) |  #Matches (Total)
-----|----------------| ---------------|-----------------|----------------|--------------- | ----------------
big_bird | 156 | 40 | 18 | 0 | 1 | 215
big_cat | 162 | 16 | 5 | 1 | 0 | 184
big_dog | 316 | 65 | 9 | 0 | 0 | 390
big_horse | 113 | 14 | 1 | 0 | 0 | 128
big_plane | 582 | 288 | 78 | 16 | 3 | 967
black_cat | 372 | 209 | 151 | 82 | 26 | 840
black_dog | 319 | 122 | 68 | 43 | 4 | 556
blue_car | 58 | 17 | 7 | 2 | 0 | 84
brown_dog | 449 | 133 | 28 | 3 | 0 | 613
eat_man | 290 | 151 | 74 | 31 | 9 | 555
fluffy_cat | 70 | 16 | 3 | 0 | 0 | 89
fly_bird | 95 | 55 | 47 | 30 | 18 | 245
old_bench | 25 | 7 | 1 | 0 | 0 | 33
old_car | 125 | 31 | 13 | 2 | 1 | 172
old_chair | 26 | 1 | 0 | 0 | 0 | 27
old_couch | 7 | 1 | 0 | 0 | 0 | 8
old_man | 255 | 43 | 13 | 3 | 0 | 314
old_table | 12 | 0 | 0 | 0 | 0 | 12
old_truck | 81 | 65 | 23 | 6 | 1 | 176
old_woman | 111 | 36 | 5 | 2 | 0 | 154
orange_cat | 172 | 87 | 36 | 6 | 1 | 302
red_bus | 225 | 160 | 125 | 49 | 7 | 566
red_car | 127 | 27 | 14 | 5 | 0 | 173
red_truck | 102 | 41 | 40 | 19 | 3 | 205
ride_woman | 359 | 129 | 71 | 30 | 6 | 595
small_cat | 196 | 41 | 13 | 2 | 0 | 252
small_plane | 272 | 116 | 59 | 24 | 10 | 481
wear_dog | 67 | 47 | 16 | 8 | 3 | 141
white_cat | 454 | 216 | 100 | 35 | 7 | 812
wooden_bench | 33 | 3 | 0 | 0 | 0 | 36
wooden_table | 14 | 0 | 0 | 0 | 0 | 14
young_man | 3231 | 870 | 243 | 31 | 3 | 4378
young_woman | 1744 | 445 | 95 | 11 | 0 | 2295

### COCO validation dataset

Pair | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5) |  #Matches (Total)  
-----|----------------| ---------------|-----------------|----------------|--------------  | ---------------- 
big_bird | 85 | 30 | 5 | 3 | 0 | 123
big_cat | 98 | 5 | 0 | 0 | 0 | 103
big_dog | 177 | 19 | 6 | 0 | 0 | 202
big_horse | 67 | 10 | 2 | 0 | 0 | 79
big_plane | 213 | 116 | 22 | 5 | 1 | 357
black_cat | 203 | 105 | 79 | 45 | 16 | 448
black_dog | 178 | 95 | 26 | 13 | 3 | 315
blue_car | 34 | 9 | 4 | 0 | 0 | 47
brown_dog | 203 | 73 | 14 | 1 | 0 | 291
eat_man | 145 | 53 | 36 | 13 | 3 | 250
fluffy_cat | 33 | 1 | 1 | 0 | 1 | 36
fly_bird | 48 | 29 | 28 | 21 | 6 | 132
old_bench | 22 | 1 | 2 | 0 | 0 | 25
old_car | 56 | 21 | 5 | 1 | 0 | 83
old_chair | 11 | 1 | 0 | 0 | 0 | 12
old_couch | 6 | 0 | 0 | 0 | 0 | 6
old_man | 123 | 24 | 13 | 4 | 2 | 166
old_table | 7 | 1 | 0 | 0 | 0 | 8
old_truck | 41 | 21 | 13 | 4 | 3 | 82
old_woman | 63 | 8 | 4 | 0 | 0 | 75
orange_cat | 95 | 46 | 15 | 6 | 0 | 162
red_bus | 111 | 66 | 32 | 18 | 5 | 232
red_car | 52 | 16 | 7 | 4 | 0 | 79
red_truck | 48 | 20 | 15 | 11 | 1 | 95
ride_woman | 185 | 64 | 30 | 16 | 5 | 300
small_cat | 120 | 23 | 5 | 1 | 0 | 149
small_plane | 90 | 30 | 31 | 6 | 1 | 158
wear_dog | 33 | 16 | 12 | 10 | 0 | 71
white_cat | 240 | 106 | 49 | 14 | 7 | 416
wooden_bench | 11 | 1 | 0 | 0 | 0 | 12
wooden_table | 7 | 0 | 0 | 0 | 0 | 7
young_man | 1550 | 460 | 93 | 17 | 1 | 2121
young_woman | 879 | 202 | 53 | 8 | 0 | 1142
