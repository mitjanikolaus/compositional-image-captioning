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
big_truck | 303 | 77 | 27 | 9 | 1 | 417
black_bird | 135 | 39 | 17 | 12 | 2 | 205
black_cat | 372 | 209 | 151 | 82 | 26 | 840
black_dog | 319 | 122 | 68 | 43 | 4 | 556
blue_boat | 98 | 28 | 10 | 2 | 0 | 138
blue_bus | 155 | 76 | 32 | 10 | 3 | 276
blue_car | 58 | 17 | 7 | 2 | 0 | 84
blue_truck | 63 | 22 | 15 | 9 | 0 | 109
brown_couch | 99 | 9 | 2 | 1 | 0 | 111
brown_dog | 449 | 133 | 28 | 3 | 0 | 613
brown_horse | 410 | 140 | 33 | 4 | 0 | 587
eat_child | 239 | 114 | 65 | 29 | 6 | 453
eat_horse | 94 | 65 | 34 | 18 | 1 | 212
eat_man | 290 | 151 | 74 | 31 | 9 | 555
eat_woman | 244 | 101 | 66 | 23 | 5 | 439
fluffy_cat | 70 | 16 | 3 | 0 | 0 | 89
fly_bird | 95 | 55 | 47 | 30 | 18 | 245
green_bench | 39 | 12 | 9 | 5 | 0 | 65
hold_child | 818 | 306 | 126 | 59 | 19 | 1328
hold_man | 2557 | 824 | 320 | 141 | 26 | 3868
hold_woman | 1459 | 560 | 264 | 110 | 27 | 2420
jump_man | 507 | 133 | 32 | 7 | 2 | 681
old_bench | 25 | 7 | 1 | 0 | 0 | 33
old_bus | 86 | 24 | 5 | 3 | 2 | 120
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
red_chair | 60 | 16 | 15 | 5 | 2 | 98
red_truck | 102 | 41 | 40 | 19 | 3 | 205
ride_child | 413 | 93 | 32 | 7 | 2 | 547
ride_woman | 359 | 129 | 71 | 30 | 6 | 595
sit_man | 1377 | 702 | 423 | 198 | 79 | 2779
small_bird | 235 | 104 | 36 | 10 | 1 | 386
small_car | 100 | 13 | 5 | 2 | 1 | 121
small_cat | 196 | 41 | 13 | 2 | 0 | 252
small_dog | 420 | 167 | 62 | 27 | 5 | 681
small_plane | 272 | 116 | 59 | 24 | 10 | 481
small_table | 225 | 30 | 6 | 0 | 0 | 261
small_woman | 576 | 275 | 174 | 48 | 6 | 1079
smile_child | 202 | 35 | 9 | 1 | 0 | 247
smile_man | 392 | 68 | 10 | 1 | 0 | 471
smile_woman | 385 | 86 | 12 | 3 | 0 | 486
stand_bird | 304 | 147 | 49 | 25 | 7 | 532
stand_child | 962 | 210 | 86 | 27 | 3 | 1288
stand_woman | 1596 | 483 | 188 | 53 | 13 | 2333
wear_child | 513 | 64 | 20 | 1 | 0 | 598
wear_dog | 67 | 47 | 16 | 8 | 3 | 141
wear_woman | 691 | 112 | 23 | 5 | 0 | 831
white_bird | 145 | 64 | 27 | 13 | 1 | 250
white_boat | 316 | 46 | 10 | 1 | 0 | 373
white_bus | 284 | 87 | 29 | 10 | 2 | 412
white_car | 136 | 31 | 11 | 5 | 0 | 183
white_cat | 454 | 216 | 100 | 35 | 7 | 812
white_dog | 381 | 164 | 56 | 17 | 4 | 622
white_horse | 151 | 57 | 36 | 17 | 3 | 264
white_table | 217 | 24 | 2 | 0 | 0 | 243
white_truck | 178 | 52 | 22 | 7 | 3 | 262
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
big_truck | 126 | 49 | 15 | 1 | 0 | 191
black_bird | 84 | 20 | 14 | 2 | 2 | 122
black_cat | 203 | 105 | 79 | 45 | 16 | 448
black_dog | 178 | 95 | 26 | 13 | 3 | 315
blue_boat | 50 | 8 | 4 | 0 | 0 | 62
blue_bus | 93 | 25 | 23 | 2 | 0 | 143
blue_car | 34 | 9 | 4 | 0 | 0 | 47
blue_truck | 32 | 15 | 8 | 3 | 1 | 59
brown_couch | 42 | 10 | 2 | 1 | 0 | 55
brown_dog | 203 | 73 | 14 | 1 | 0 | 291
brown_horse | 204 | 60 | 23 | 6 | 0 | 293
eat_child | 98 | 47 | 34 | 12 | 5 | 196
eat_horse | 42 | 33 | 24 | 7 | 0 | 106
eat_man | 145 | 53 | 36 | 13 | 3 | 250
eat_woman | 113 | 56 | 24 | 17 | 2 | 212
fluffy_cat | 33 | 1 | 1 | 0 | 1 | 36
fly_bird | 48 | 29 | 28 | 21 | 6 | 132
green_bench | 20 | 8 | 5 | 3 | 1 | 37
hold_child | 393 | 153 | 78 | 28 | 12 | 664
hold_man | 1221 | 433 | 167 | 65 | 22 | 1908
hold_woman | 716 | 288 | 128 | 61 | 12 | 1205
jump_man | 219 | 46 | 9 | 3 | 0 | 277
old_bench | 22 | 1 | 2 | 0 | 0 | 25
old_bus | 35 | 12 | 4 | 1 | 0 | 52
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
red_chair | 34 | 5 | 7 | 0 | 0 | 46
red_truck | 48 | 20 | 15 | 11 | 1 | 95
ride_child | 187 | 65 | 22 | 3 | 3 | 280
ride_woman | 185 | 64 | 30 | 16 | 5 | 300
sit_man | 696 | 346 | 210 | 117 | 40 | 1409
small_bird | 140 | 58 | 23 | 7 | 0 | 228
small_car | 48 | 6 | 2 | 2 | 0 | 58
small_cat | 120 | 23 | 5 | 1 | 0 | 149
small_dog | 205 | 69 | 25 | 15 | 2 | 316
small_plane | 90 | 30 | 31 | 6 | 1 | 158
small_table | 115 | 15 | 3 | 1 | 0 | 134
small_woman | 308 | 144 | 79 | 30 | 7 | 568
smile_child | 105 | 28 | 5 | 0 | 0 | 138
smile_man | 202 | 43 | 3 | 0 | 0 | 248
smile_woman | 163 | 38 | 9 | 1 | 0 | 211
stand_bird | 152 | 68 | 18 | 15 | 7 | 260
stand_child | 395 | 111 | 48 | 18 | 5 | 577
stand_woman | 804 | 218 | 81 | 32 | 7 | 1142
wear_child | 252 | 48 | 8 | 1 | 0 | 309
wear_dog | 33 | 16 | 12 | 10 | 0 | 71
wear_woman | 341 | 52 | 9 | 0 | 0 | 402
white_bird | 95 | 32 | 11 | 4 | 2 | 144
white_boat | 157 | 33 | 6 | 0 | 0 | 196
white_bus | 164 | 45 | 17 | 1 | 0 | 227
white_car | 60 | 14 | 5 | 3 | 0 | 82
white_cat | 240 | 106 | 49 | 14 | 7 | 416
white_dog | 200 | 79 | 19 | 6 | 2 | 306
white_horse | 84 | 27 | 27 | 10 | 3 | 151
white_table | 83 | 8 | 1 | 0 | 0 | 92
white_truck | 88 | 26 | 5 | 2 | 0 | 121
wooden_bench | 11 | 1 | 0 | 0 | 0 | 12
wooden_table | 7 | 0 | 0 | 0 | 0 | 7
young_man | 1550 | 460 | 93 | 17 | 1 | 2121
young_woman | 879 | 202 | 53 | 8 | 0 | 1142
