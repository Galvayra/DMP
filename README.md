# DeepPredict
Deep Learning for mortality

step 1) Parsing Dataset to erase noise

	1-1) set input or ouput name 
	Useage) python parsing.py -input INPUT -output OUTPUT
	
	1-2) set target symptom (parsing data where the target symptom is True)
	Useage) python parsing.py -target s(sepsis) 
		
	{s=sepsis, p=pneumonia, b=bacteremia}


step 2) Encoding Dataset to make vector (vectorization)
	
	2-1) set input or ouput name
	Useage) python encoding.py -input INPUT -output OUTPUT

	2-2) set target symptom (when data is vetorized, the target symptom is excepted)
	Useage) python encoding.py -target s(sepsis)

	2-3) set k-fold (default is 5)
	Useage) python encoding.py -fold 1


*) intergrate Step 1&2, use a shell script

*-1) do parsing then encoding
Useage) ./build.sh --parsing 1 --all 1

*-2) do encoding k-fold of just specific symptoms 
Useage) ./build.sh --fold 1

*-3) If you want to make vector apply specific columns, annotate the lines in dataset/variables.py
Useage) ./build.sh --id ID
