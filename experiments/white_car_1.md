# Case study
Image COCO ID: 367893

## Attention
![Visualized Attention](white_car_attention_1.png)

## Decoding Beam (k=5)
```

['<start>', 'a'] 				 Score: -0.2757272720336914
['<start>', 'the'] 				 Score: -2.839357852935791
['<start>', 'two'] 				 Score: -3.048694610595703
['<start>', 'an'] 				 Score: -3.136408805847168
['<start>', 'there'] 				 Score: -3.78116512298584


['<start>', 'a', 'red'] 				 Score: -1.140181541442871
['<start>', 'a', 'truck'] 				 Score: -2.5694539546966553
['<start>', 'a', 'large'] 				 Score: -3.5861520767211914
['<start>', 'a', 'small'] 				 Score: -3.608610153198242
['<start>', 'a', 'white'] 				 Score: -3.6584582328796387


['<start>', 'a', 'red', 'truck'] 				 Score: -1.46600341796875
['<start>', 'a', 'red', 'pickup'] 				 Score: -3.5245726108551025
['<start>', 'a', 'red', 'pick'] 				 Score: -3.7986927032470703
['<start>', 'a', 'small', 'red'] 				 Score: -3.830996036529541
['<start>', 'a', 'large', 'red'] 				 Score: -3.9535303115844727


['<start>', 'a', 'red', 'truck', 'parked'] 				 Score: -2.429969072341919
['<start>', 'a', 'red', 'truck', 'with'] 				 Score: -3.3345818519592285
['<start>', 'a', 'red', 'pickup', 'truck'] 				 Score: -3.532701253890991
['<start>', 'a', 'red', 'truck', 'is'] 				 Score: -3.537292003631592
['<start>', 'a', 'red', 'pick', 'up'] 				 Score: -3.8025755882263184


['<start>', 'a', 'red', 'truck', 'parked', 'on'] 				 Score: -3.287463426589966
['<start>', 'a', 'red', 'truck', 'parked', 'in'] 				 Score: -3.6330294609069824
['<start>', 'a', 'red', 'pick', 'up', 'truck'] 				 Score: -3.814617156982422
['<start>', 'a', 'red', 'truck', 'with', 'a'] 				 Score: -3.843029499053955
['<start>', 'a', 'red', 'truck', 'is', 'parked'] 				 Score: -3.9162116050720215


['<start>', 'a', 'red', 'truck', 'parked', 'on', 'a'] 				 Score: -3.940936326980591
['<start>', 'a', 'red', 'truck', 'parked', 'in', 'a'] 				 Score: -4.161174774169922
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the'] 				 Score: -4.178128719329834
['<start>', 'a', 'red', 'truck', 'is', 'parked', 'on'] 				 Score: -4.9346818923950195
['<start>', 'a', 'red', 'pick', 'up', 'truck', 'parked'] 				 Score: -4.980012893676758


['<start>', 'a', 'red', 'truck', 'parked', 'in', 'a', 'parking'] 				 Score: -4.216998100280762
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side'] 				 Score: -4.306332588195801
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'a', 'city'] 				 Score: -4.940716743469238
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'a', 'street'] 				 Score: -5.253949165344238
['<start>', 'a', 'red', 'truck', 'is', 'parked', 'on', 'a'] 				 Score: -5.628119468688965


['<start>', 'a', 'red', 'truck', 'parked', 'in', 'a', 'parking', 'lot'] 				 Score: -4.293420791625977
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of'] 				 Score: -4.307692527770996
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'a', 'city', 'street'] 				 Score: -4.989635944366455
['<start>', 'a', 'red', 'truck', 'is', 'parked', 'on', 'a', 'street'] 				 Score: -6.674820423126221
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'a', 'street', 'with'] 				 Score: -6.708817481994629


['<start>', 'a', 'red', 'truck', 'parked', 'in', 'a', 'parking', 'lot', '<end>'] 				 Score: -4.527951717376709
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'a'] 				 Score: -4.6375555992126465
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'a', 'city', 'street', '<end>'] 				 Score: -5.053534984588623
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'the'] 				 Score: -5.624790668487549
['<start>', 'a', 'red', 'truck', 'parked', 'in', 'a', 'parking', 'lot', 'next'] 				 Score: -7.175141334533691


['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'a', 'road'] 				 Score: -5.305601596832275
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'a', 'street'] 				 Score: -5.4472455978393555
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'the', 'road'] 				 Score: -6.122622966766357


['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'a', 'road', '<end>'] 				 Score: -5.387211799621582
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'a', 'street', '<end>'] 				 Score: -5.5092878341674805
['<start>', 'a', 'red', 'truck', 'parked', 'on', 'the', 'side', 'of', 'the', 'road', '<end>'] 				 Score: -6.187585353851318
```
