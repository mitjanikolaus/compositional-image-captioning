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
brown_dog | 615.0 | 164.0 | 31.0 | 3.0 | 0.0 | 
eat_man | 553.0 | 262.0 | 106.0 | 39.0 | 8.0 | 
green_bench | 65.0 | 26.0 | 14.0 | 5.0 | 0.0 | 
red_chair | 90.0 | 35.0 | 18.0 | 7.0 | 2.0 | 
ride_woman | 591.0 | 234.0 | 107.0 | 35.0 | 6.0 | 
sit_cat | 1385.0 | 839.0 | 519.0 | 285.0 | 90.0 | 
small_plane | 488.0 | 209.0 | 94.0 | 35.0 | 10.0 | 
white_car | 839.0 | 273.0 | 101.0 | 31.0 | 5.0 | 
wooden_table | 200.0 | 12.0 | 1.0 | 0.0 | 0.0 |

Noun | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
car | 7408.0 | 5575.0 | 4669.0 | 3949.0 | 2821.0 | 
dog | 3625.0 | 3281.0 | 3010.0 | 2643.0 | 2063.0 | 
cat | 2858.0 | 2723.0 | 2650.0 | 2562.0 | 2224.0 | 
bench | 1921.0 | 1495.0 | 1325.0 | 1158.0 | 841.0 | 
table | 22432.0 | 10615.0 | 5980.0 | 3363.0 | 1387.0 | 
chair | 2794.0 | 1443.0 | 867.0 | 513.0 | 256.0 | 
woman | 11979.0 | 9268.0 | 7538.0 | 5634.0 | 2953.0 | 
man | 23870.0 | 18148.0 | 13599.0 | 9146.0 | 4317.0 | 
plane | 2392.0 | 2236.0 | 2182.0 | 2055.0 | 1524.0 | 

Adjective | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
white | 16709.0 | 5476.0 | 1759.0 | 479.0 | 85.0 | 
brown | 4336.0 | 1030.0 | 281.0 | 64.0 | 11.0 | 
small | 13780.0 | 4225.0 | 1424.0 | 405.0 | 64.0 | 
wooden | 4497.0 | 1245.0 | 385.0 | 101.0 | 18.0 | 
green | 6013.0 | 1710.0 | 625.0 | 209.0 | 51.0 | 
red | 7147.0 | 2870.0 | 1259.0 | 464.0 | 108.0 | 

Verb | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| --------------- |----------------|----------------|--------------
sit | 25042.0 | 11491.0 | 5653.0 | 2664.0 | 910.0 | 
ride | 7605.0 | 3869.0 | 2130.0 | 1011.0 | 296.0 | 
eat | 4384.0 | 2136.0 | 1293.0 | 758.0 | 325.0 | 


### COCO validation dataset

Pair | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| ---------------|-----------------|----------------|--------------
brown_dog | 290.0 | 87.0 | 15.0 | 1.0 | 0.0 | 
eat_man | 247.0 | 103.0 | 50.0 | 16.0 | 3.0 | 
green_bench | 37.0 | 17.0 | 9.0 | 4.0 | 1.0 | 
red_chair | 39.0 | 10.0 | 6.0 | 0.0 | 0.0 | 
ride_woman | 298.0 | 115.0 | 51.0 | 18.0 | 3.0 | 
sit_cat | 775.0 | 479.0 | 277.0 | 147.0 | 57.0 | 
small_plane | 159.0 | 67.0 | 38.0 | 7.0 | 2.0 | 
white_car | 419.0 | 126.0 | 38.0 | 7.0 | 0.0 | 
wooden_table | 116.0 | 13.0 | 2.0 | 1.0 | 0.0 | 


Noun | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
car | 3594.0 | 2691.0 | 2246.0 | 1883.0 | 1369.0 | 
dog | 1834.0 | 1678.0 | 1542.0 | 1376.0 | 1067.0 | 
cat | 1495.0 | 1431.0 | 1393.0 | 1355.0 | 1213.0 | 
bench | 1030.0 | 797.0 | 700.0 | 604.0 | 466.0 | 
table | 10899.0 | 5128.0 | 2860.0 | 1569.0 | 632.0 | 
chair | 1381.0 | 755.0 | 465.0 | 266.0 | 126.0 | 
woman | 5752.0 | 4369.0 | 3489.0 | 2621.0 | 1385.0 | 
man | 11314.0 | 8637.0 | 6490.0 | 4356.0 | 2056.0 | 
plane | 903.0 | 839.0 | 809.0 | 761.0 | 538.0 | 

Adjective | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-------|----------------| ---------------|-----------------|----------------|--------------
white | 8188.0 | 2647.0 | 835.0 | 220.0 | 33.0 | 
brown | 2192.0 | 512.0 | 146.0 | 32.0 | 1.0 | 
small | 6733.0 | 2026.0 | 687.0 | 210.0 | 32.0 | 
wooden | 2149.0 | 615.0 | 182.0 | 45.0 | 6.0 | 
green | 3024.0 | 883.0 | 315.0 | 114.0 | 25.0 | 
red | 3570.0 | 1385.0 | 610.0 | 236.0 | 50.0 | 

Verb | #Matches (N=1) |  #Matches (N=2) | #Matches (N=3) | #Matches (N=4) | #Matches (N=5)
-----|----------------| --------------- |----------------|----------------|--------------
sit | 12499.0 | 5791.0 | 2778.0 | 1307.0 | 474.0 | 
ride | 3699.0 | 1920.0 | 1031.0 | 459.0 | 135.0 | 
eat | 2077.0 | 1018.0 | 606.0 | 344.0 | 146.0 | 


