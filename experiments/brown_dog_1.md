# Case study
Image COCO ID: 537955

## Attention
![Visualized Attention](brown_dog_attention_1.png)

## Decoding Beam (k=5)
```
['<start>', 'a'] 				 Score: -0.17604684829711914
['<start>', 'the'] 				 Score: -3.1701157093048096
['<start>', 'two'] 				 Score: -3.6444098949432373
['<start>', 'an'] 				 Score: -3.7512166500091553
['<start>', 'there'] 				 Score: -4.348722457885742


['<start>', 'a', 'dog'] 				 Score: -0.6014971733093262
['<start>', 'a', 'large'] 				 Score: -3.196495294570923
['<start>', 'a', 'small'] 				 Score: -3.397876262664795
['<start>', 'the', 'dog'] 				 Score: -3.641566514968872
['<start>', 'two', 'dogs'] 				 Score: -3.712994337081909


['<start>', 'a', 'dog', 'laying'] 				 Score: -1.9677605628967285
['<start>', 'a', 'dog', 'is'] 				 Score: -2.216230869293213
['<start>', 'a', 'dog', 'sitting'] 				 Score: -3.299713134765625
['<start>', 'a', 'dog', 'that'] 				 Score: -3.369974374771118
['<start>', 'a', 'small', 'dog'] 				 Score: -3.4425692558288574


['<start>', 'a', 'dog', 'laying', 'on'] 				 Score: -2.231490135192871
['<start>', 'a', 'dog', 'is', 'laying'] 				 Score: -3.1427464485168457
['<start>', 'a', 'dog', 'that', 'is'] 				 Score: -3.4603168964385986
['<start>', 'a', 'dog', 'sitting', 'on'] 				 Score: -3.6427736282348633
['<start>', 'a', 'dog', 'is', 'sitting'] 				 Score: -4.049251079559326


['<start>', 'a', 'dog', 'laying', 'on', 'a'] 				 Score: -2.826462745666504
['<start>', 'a', 'dog', 'laying', 'on', 'the'] 				 Score: -3.5578770637512207
['<start>', 'a', 'dog', 'is', 'laying', 'on'] 				 Score: -3.678433895111084
['<start>', 'a', 'dog', 'that', 'is', 'laying'] 				 Score: -3.8408052921295166
['<start>', 'a', 'dog', 'sitting', 'on', 'a'] 				 Score: -4.154476642608643


['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor'] 				 Score: -3.71290922164917
['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down'] 				 Score: -4.212189674377441
['<start>', 'a', 'dog', 'is', 'laying', 'on', 'a'] 				 Score: -4.339470863342285
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden'] 				 Score: -4.375129699707031
['<start>', 'a', 'dog', 'is', 'laying', 'on', 'the'] 				 Score: -4.687376499176025


['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor'] 				 Score: -4.460048675537109
['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down', 'on'] 				 Score: -4.831249713897705
['<start>', 'a', 'dog', 'is', 'laying', 'on', 'the', 'floor'] 				 Score: -4.846985340118408
['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'next'] 				 Score: -5.172623157501221
['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'in'] 				 Score: -5.453830718994141


['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down', 'on', 'a'] 				 Score: -5.112282752990723
['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'next', 'to'] 				 Score: -5.173093318939209
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'next'] 				 Score: -5.895857334136963
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'in'] 				 Score: -6.0718865394592285
['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'in', 'a'] 				 Score: -6.2036590576171875


['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'next', 'to', 'a'] 				 Score: -5.244523525238037
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'next', 'to'] 				 Score: -5.896351337432861
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'in', 'a'] 				 Score: -6.666940689086914
['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down', 'on', 'a', 'rug'] 				 Score: -6.7408552169799805
['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down', 'on', 'a', 'floor'] 				 Score: -7.008491039276123


['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'next', 'to', 'a'] 				 Score: -5.997666835784912
['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down', 'on', 'a', 'rug', '<end>'] 				 Score: -6.830450057983398
['<start>', 'a', 'dog', 'that', 'is', 'laying', 'down', 'on', 'a', 'floor', '<end>'] 				 Score: -7.129322528839111
['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'next', 'to', 'a', 'wooden'] 				 Score: -7.245656490325928
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'in', 'a', 'kitchen'] 				 Score: -7.775879383087158


['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'in', 'a', 'kitchen', '<end>'] 				 Score: -7.879330635070801
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'next', 'to', 'a', 'wooden'] 				 Score: -8.050020217895508
['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'next', 'to', 'a', 'wooden', 'floor'] 				 Score: -8.07992172241211


['<start>', 'a', 'dog', 'laying', 'on', 'the', 'floor', 'next', 'to', 'a', 'wooden', 'floor', '<end>'] 				 Score: -8.155442237854004
['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'next', 'to', 'a', 'wooden', 'floor'] 				 Score: -8.760151863098145


['<start>', 'a', 'dog', 'laying', 'on', 'a', 'wooden', 'floor', 'next', 'to', 'a', 'wooden', 'floor', '<end>'] 				 Score: -8.78670883178711
```

## Scores for alternatives of the top 5 sentences

These examples show the scores for artificially created sentences that incorporate descriptions of colour. 

```
a dog that is laying down on a rug 	        Score: -6.83044958114624
a brown dog that is laying down on a rug 	Score: -12.176456451416016
a black dog that is laying down on a rug 	Score: -11.864744186401367
a white dog that is laying down on a rug 	Score: -14.034316062927246
a blue dog that is laying down on a rug 	Score: -17.353790283203125
a red dog that is laying down on a rug 	    Score: -14.830488204956055
```
