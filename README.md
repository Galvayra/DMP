# DeepPredict




### Training medical data

First of all, import this project using ~/.bash_profile


step 1) Parsing Dataset to erase noise for split train, validation, and test set

	1-1) set input or output name
	Useage) python parsing.py -input INPUT -output OUTPUT
	(path of dir,  "input=dataset/origin/"  "output=dataset/parsing/")
	
	1-2) set target symptom (parsing data where the target symptom is True)
	Useage) python parsing.py -target s (sepsis) 
	{s=sepsis, p=pneumonia, b=bacteremia}

	You can use sampling or set ratio to divide training, valid, and test set



step 2) Encoding Dataset to make vector (vectorization)
	
	==2-1) Vectorization

	2-1-1) set input or ouput name, you can use word2vec or not
	Useage) python encoding.py -input INPUT -output OUTPUT -ver 1 -w2v 0
	(path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/",  "w2v file=modeling/embedding/")

	2-1-2) set target symptom (when data is vetorized, the target symptom is excepted)
	Useage) python encoding.py -target s (sepsis) -ver 1


	==2-2) Feature Selection using Random Forest

    2-2-1) if you want to select important features, use build_feature_selection.sh
    Useage) ./build_feature_selection.sh --input INPUT --output OUTPUT --fs_name FS_NAME --ntree NTREE
    (path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/", "fs_name=modeling/fsResult/")
    
	2-2-2) Finally, make a vector using "fs" option
	Useage) python encoding.py -input INPUT -output OUTPUT -fs FS -n_feature N -w2v 0
	(path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/")
	(              "fs=modeling/fsResult/"  "w2v file=modeling/embedding/")



Step #) Convert vector to image for training Convolutional Neural Network

	#) copy ct images for training (This is not useful script)
	./build_ct_images.sh 

	##) convert vector to image
	Useage) python convert_images.py -vector VECTOR -output OUTPUT -resize R
	(path of dir,  "vector=modeling/vectors/"  "output=dataset/images/image_data/"
	
	A structure of directories
	version 1 = alive and death (Default, for training this system)
	version 2 = train, valid, and test (training another architecture)
	
	
 
Step 3) Training

    It can be trained to FFNN or CNN (by tensorflow)
    
    
	3-1) training vector to Neural Network (If you want show options, use -h)
	Useage) python training.py -model (ffnn|cnn) -image_dir I -vector VECTOR -epoch E -hidden H -learn L -tensor_dir TENSOR -delete D -show S -ver V 
	(path of dir,  "vector=modeling/vectors/"  "tensor_dir=logs/ & modeling/save/")
	(option,  I = set a path of images for CNN (It is necessary when you use CNN))
	(         TENSOR = save a path of log for tensorboard and tensor for loading)
	(         D = set a delete log or tensor directory (This is useful when you retrain and rewrite))
	(         S = show parameters when the model trained in NN)
	(         V = select a training version)
	
	Training Version (It is same in predict script)
	version 1 = k cross validation
	version 2 = optimize hyper-parameter using training, validation, and test set
	
	

	#) if you want to train and test at once, use run.sh
	Useage) ./run.sh --model M --image_dir I --vector VECTOR --feature F --target T --epoch E --hidden H --learn L --tensor_dir TENSOR --delete D --save SAVE --plot P --show S --ver V
	
	

Step 4) Get a performance


    4-1) predict and get a performance by trained model (default is svm) (If you wnat show options, use -h)
	Useage) python predict.py -model (svm|ffnn|cnn) -image_dir I -vector VECTOR -tensor_dir TENSOR -save SAVE -plot P -show S -ver V 
	(path of dir,  "vector=modeling/vectors/"  "tensor_dir=modeling/save/"  "save=analysis/")
	(option,  I = set a path of images for CNN (It is necessary when you use CNN))
	(         LOG = load a path of tensor)
	(         SAVE = save a performance of system using csv file)
	(         P = show a ROC curve of model)
	(         S = show parameters when the model trained in NN)
	(         V = select a training version)



### Training CT Images



Step 1) Parsing Dataset to erase noise for split train, validation, and test set

    For getting dataset which consist of patients who get a ct image
    Useage) python parsing.py -input INPUT -output OUTPUT -parsing_image 1
	(path of dir,  "input=dataset/origin/"  "output=dataset/parsing/")



step 2) Encoding Dataset to make vector (vectorization)
	Use encode_image option, it will be making tfRecord and pickle of ct images
	Useage) python encoding.py -input INPUT -output OUTPUT -encode_image 1 -w2v 0
	(path of dir,  "input=dataset/parsing/"  "output=modeling/vectors/",  "w2v file=modeling/embedding/")



Step 2) Training

    You can train images using transfer learning or training CNN model
    
    If you want to transfer learning, set a "IMAGE_SIZE" variable to "what you want to set image size"
    If you want to training CNN model, set a "IMAGE_SIZE" variable to "INITIAL_IMAGE_SIZE"
    The "IMAGE_SIZE" variable is in learning/variables.py
    
    Useage) python fine_tuning.py -model (transfer|tuning|ffnn|cnn) -vector VECTOR -epoch E -hidden H -learn L -tensor_dir TENSOR -delete D -show S -ver V
	(path of dir,  "vector=modeling/vectors/"  "tensor_dir=logs/ & modeling/save/")
	(option,  TENSOR = save a path of log for tensorboard and tensor for loading)
	(         D = set a delete log or tensor directory (This is useful when you retrain and rewrite))
	(         S = show parameters when the model trained in NN)
	(         V = select a training version)
