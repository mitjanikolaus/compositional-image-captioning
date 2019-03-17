# Case study
Image COCO ID: 553482

## Attention
![Visualized Attention](white_car_attention_2.png)

## Decoding Beam (k=5)
```

['<start>', 'a'] 				 Score: -0.3174123764038086
['<start>', 'two'] 				 Score: -2.5163216590881348
['<start>', 'the'] 				 Score: -2.9138708114624023
['<start>', 'an'] 				 Score: -3.552842140197754
['<start>', 'there'] 				 Score: -3.8306007385253906


['<start>', 'a', 'truck'] 				 Score: -1.8993024826049805
['<start>', 'a', 'large'] 				 Score: -2.822093963623047
['<start>', 'a', 'semi'] 				 Score: -3.2751426696777344
['<start>', 'a', 'man'] 				 Score: -3.4460272789001465
['<start>', 'a', 'couple'] 				 Score: -3.5908684730529785


['<start>', 'a', 'large', 'truck'] 				 Score: -3.1770973205566406
['<start>', 'a', 'semi', 'truck'] 				 Score: -3.4440207481384277
['<start>', 'a', 'truck', 'with'] 				 Score: -3.5629382133483887
['<start>', 'a', 'couple', 'of'] 				 Score: -3.620218276977539
['<start>', 'a', 'truck', 'is'] 				 Score: -3.9223222732543945


['<start>', 'a', 'truck', 'with', 'a'] 				 Score: -4.101783275604248
['<start>', 'a', 'couple', 'of', 'trucks'] 				 Score: -4.296609401702881
['<start>', 'a', 'large', 'truck', 'with'] 				 Score: -4.8500165939331055
['<start>', 'a', 'semi', 'truck', 'is'] 				 Score: -4.979515075683594
['<start>', 'a', 'large', 'truck', 'is'] 				 Score: -5.015174865722656


['<start>', 'a', 'large', 'truck', 'with', 'a'] 				 Score: -5.423736572265625
['<start>', 'a', 'couple', 'of', 'trucks', 'are'] 				 Score: -5.583391189575195
['<start>', 'a', 'couple', 'of', 'trucks', 'parked'] 				 Score: -5.975870609283447
['<start>', 'a', 'semi', 'truck', 'is', 'parked'] 				 Score: -6.069820880889893
['<start>', 'a', 'truck', 'with', 'a', 'crane'] 				 Score: -6.12136173248291


['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked'] 				 Score: -6.135281562805176
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on'] 				 Score: -6.9564385414123535
['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next'] 				 Score: -7.081141948699951
['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'on'] 				 Score: -7.255927085876465
['<start>', 'a', 'truck', 'with', 'a', 'crane', 'on'] 				 Score: -7.316641807556152


['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to'] 				 Score: -7.082543849945068
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on'] 				 Score: -7.277476787567139
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'in'] 				 Score: -7.439879417419434
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on', 'the'] 				 Score: -7.4482879638671875
['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'on', 'a'] 				 Score: -7.741931915283203


['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each'] 				 Score: -7.110569477081299
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on', 'the', 'side'] 				 Score: -7.74889612197876
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'in', 'a'] 				 Score: -7.765522003173828
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'the'] 				 Score: -7.777238845825195
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'a'] 				 Score: -8.291178703308105


['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each', 'other'] 				 Score: -7.110934734344482
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on', 'the', 'side', 'of'] 				 Score: -7.756710052490234
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'in', 'a', 'parking'] 				 Score: -8.181280136108398
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'the', 'side'] 				 Score: -8.187273025512695
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'a', 'street'] 				 Score: -9.402862548828125


['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each', 'other', 'on'] 				 Score: -8.05058765411377
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'in', 'a', 'parking', 'lot'] 				 Score: -8.19157886505127
['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each', 'other', '<end>'] 				 Score: -8.203134536743164
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'the', 'side', 'of'] 				 Score: -8.207466125488281
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on', 'the', 'side', 'of', 'the'] 				 Score: -8.443843841552734


['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'in', 'a', 'parking', 'lot', '<end>'] 				 Score: -8.212977409362793
['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each', 'other', 'on', 'a'] 				 Score: -8.219573974609375
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'the', 'side', 'of', 'the'] 				 Score: -8.59754753112793
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on', 'the', 'side', 'of', 'the', 'road'] 				 Score: -8.738760948181152


['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each', 'other', 'on', 'a', 'street'] 				 Score: -8.724344253540039
['<start>', 'a', 'semi', 'truck', 'is', 'parked', 'on', 'the', 'side', 'of', 'the', 'road', '<end>'] 				 Score: -8.824304580688477
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'the', 'side', 'of', 'the', 'road'] 				 Score: -8.887423515319824


['<start>', 'a', 'couple', 'of', 'trucks', 'parked', 'next', 'to', 'each', 'other', 'on', 'a', 'street', '<end>'] 				 Score: -8.730511665344238
['<start>', 'a', 'couple', 'of', 'trucks', 'are', 'parked', 'on', 'the', 'side', 'of', 'the', 'road', '<end>'] 				 Score: -8.92381477355957
```
