# captioning-models data

Captions are counted as matches also when one of the synonyms for the respective adjective or nouns is present (see the
respective JSON files for a list of synonyms considered). For adjective-noun pair matches, we only consider captions
where the adjective is also describing the target noun (using the
[stanfordnlp dependency parser](https://github.com/stanfordnlp/stanfordnlp)).

'N' denotes the agreement among the different captions, e.g. 
'N=3' means that 3 of the 5 image captions contain the target.

## Numbers of matching adjective-noun or verb-noun pairs

### COCO training dataset

Pair | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| ---------------|-----------------|----------------|--------------
brown_dog | 451.0 | 133.0 | 28.0 | 3.0 | 0.0 | 
eat_man | 291.0 | 156.0 | 67.0 | 31.0 | 8.0 | 
green_bench | 39.0 | 12.0 | 9.0 | 5.0 | 0.0 | 
red_chair | 55.0 | 17.0 | 11.0 | 5.0 | 2.0 | 
ride_woman | 357.0 | 127.0 | 72.0 | 29.0 | 6.0 | 
sit_cat | 546.0 | 320.0 | 234.0 | 195.0 | 90.0 | 
small_plane | 279.0 | 115.0 | 59.0 | 25.0 | 10.0 | 
white_car | 566.0 | 172.0 | 70.0 | 26.0 | 5.0 | 
wooden_table | 188.0 | 11.0 | 1.0 | 0.0 | 0.0 | 

Noun | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
dog | 344.0 | 271.0 | 367.0 | 580.0 | 2063.0 | 
man | 5722.0 | 4549.0 | 4453.0 | 4829.0 | 4317.0 | 
bench | 426.0 | 170.0 | 167.0 | 317.0 | 841.0 | 
chair | 1351.0 | 576.0 | 354.0 | 257.0 | 256.0 | 
woman | 2711.0 | 1730.0 | 1904.0 | 2681.0 | 2953.0 | 
cat | 135.0 | 73.0 | 88.0 | 338.0 | 2224.0 | 
plane | 156.0 | 54.0 | 127.0 | 531.0 | 1524.0 | 
car | 1833.0 | 906.0 | 720.0 | 1128.0 | 2821.0 | 
table | 11817.0 | 4635.0 | 2617.0 | 1976.0 | 1387.0 | 

Adjective | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
brown | 3306.0 | 749.0 | 217.0 | 53.0 | 11.0 | 
green | 4303.0 | 1085.0 | 416.0 | 158.0 | 51.0 | 
red | 4277.0 | 1611.0 | 795.0 | 356.0 | 108.0 | 
small | 9555.0 | 2801.0 | 1019.0 | 341.0 | 64.0 | 
white | 11233.0 | 3717.0 | 1280.0 | 394.0 | 85.0 | 
wooden | 3252.0 | 860.0 | 284.0 | 83.0 | 18.0 | 

Verb | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| --------------- |----------------|----------------|--------------
eat | 2248.0 | 843.0 | 535.0 | 433.0 | 325.0 | 
ride | 3736.0 | 1739.0 | 1119.0 | 715.0 | 296.0 | 
sit | 13551.0 | 5838.0 | 2989.0 | 1754.0 | 910.0 | 


### COCO validation dataset

Pair | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| ---------------|-----------------|----------------|--------------
brown_dog | 203.0 | 72.0 | 14.0 | 1.0 | 0.0 | 
eat_man | 144.0 | 53.0 | 34.0 | 13.0 | 3.0 | 
green_bench | 20.0 | 8.0 | 5.0 | 3.0 | 1.0 | 
red_chair | 29.0 | 4.0 | 6.0 | 0.0 | 0.0 | 
ride_woman | 183.0 | 64.0 | 33.0 | 15.0 | 3.0 | 
sit_cat | 296.0 | 202.0 | 130.0 | 90.0 | 57.0 | 
small_plane | 92.0 | 29.0 | 31.0 | 5.0 | 2.0 | 
white_car | 293.0 | 88.0 | 31.0 | 7.0 | 0.0 | 
wooden_table | 103.0 | 11.0 | 1.0 | 1.0 | 0.0 | 


Noun | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
dog | 156.0 | 136.0 | 166.0 | 309.0 | 1067.0 | 
man | 2677.0 | 2147.0 | 2134.0 | 2300.0 | 2056.0 | 
bench | 233.0 | 97.0 | 96.0 | 138.0 | 466.0 | 
chair | 626.0 | 290.0 | 199.0 | 140.0 | 126.0 | 
woman | 1383.0 | 880.0 | 868.0 | 1236.0 | 1385.0 | 
cat | 64.0 | 38.0 | 38.0 | 142.0 | 1213.0 | 
plane | 64.0 | 30.0 | 48.0 | 223.0 | 538.0 | 
car | 903.0 | 445.0 | 363.0 | 514.0 | 1369.0 | 
table | 5771.0 | 2268.0 | 1291.0 | 937.0 | 632.0 | 

Adjective | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
brown | 1680.0 | 366.0 | 114.0 | 31.0 | 1.0 | 
green | 2141.0 | 568.0 | 201.0 | 89.0 | 25.0 | 
red | 2185.0 | 775.0 | 374.0 | 186.0 | 50.0 | 
small | 4707.0 | 1339.0 | 477.0 | 178.0 | 32.0 | 
white | 5541.0 | 1812.0 | 615.0 | 187.0 | 33.0 | 
wooden | 1534.0 | 433.0 | 137.0 | 39.0 | 6.0 | 

Verb | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| --------------- |----------------|----------------|--------------
eat | 1059.0 | 412.0 | 262.0 | 198.0 | 146.0 | 
ride | 1779.0 | 889.0 | 572.0 | 324.0 | 135.0 | 
sit | 6708.0 | 3013.0 | 1471.0 | 833.0 | 474.0 | 


