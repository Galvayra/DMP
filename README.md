# DeepPredict
Deep Learning for mortality

step 1) Parsing Dataset to erase noise

	1-1) set input or ouput name
	Useage) python parsing.py -input INPUT -output OUTPUT
	(path of dir,  "input=dataset/origin/"  "output=dataset/parsing/")
	
	1-2) set target symptom (parsing data where the target symptom is True)
	Useage) python parsing.py -target s (sepsis) 
	{s=sepsis, p=pneumonia, b=bacteremia}

	1-3) you can use sampling or set ratio to divide training, valid, and test set


step 2) Encoding Dataset to make vector (vectorization)
	
	==2-1) Vectorization

	2-1-1) set input or ouput name, you can use word2vec or not
	Useage) python encoding.py -input INPUT -output OUTPUT -ver 1 -w2v 0
	(path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/")
	(              "w2v file=modeling/embedding/")

	2-1-2) set target symptom (when data is vetorized, the target symptom is excepted)
	Useage) python encoding.py -target s (sepsis) -ver 1



	==2-2) Feature Selection using Random Forest

	2-2-1) if you want to select important features, use "ver 2" option 
	Useage) python encoding.py -input INPUT -output OUTPUT -ver 2 
	(path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/")

	2-2-2) you make a log file to save importance of features
	Useage) python extract_feature.py -vector VECTOR -output OUTPUT -ntree N
	(path of dir,  "vector=modeling/vectors/"  "output=modeling/fsResult/")

	2-2-3) Finally, make a vector using "fs" option
	Useage) python encoding.py -input INPUT -output OUTPUT -fs FS -n_feature N -w2v 0
	(path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/")
	(              "fs=modeling/fsResult/"  "w2v file=modeling/embedding/")



Step #) Convert vector to image for training Convolutional Neural Network

	#) copy ct images for training (This is not useful script)
	./build_ct_images.sh 

	#) convert vector to image
	Useage) python convert_images.py -vector VECTOR -output OUTPUT -resize R
	(path of dir,  "vector=modeling/vectors/"  "output=dataset/images/image_data/"
	
	A structure of directories
	version 1 = alive and death (Default, for training this system)
	version 2 = train, valid, and test (training another architecture)
	
	
 
Step 3) Training

Step 4) Get a performance
