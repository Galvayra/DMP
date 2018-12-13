## train ffnn
#/bin/bash

# minimal test

for vector in $@
do
	echo "test" $vector

	./run.sh --vector modeling/vectors/$vector --log ffnn_h1_$vector --model ffnn --epoch 5000 --learn 0.00002 --hidden 1 --result ffnn_h1_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log ffnn_h2_$vector --model ffnn --epoch 5000 --learn 0.00002 --hidden 2 --result ffnn_h2_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log ffnn_h3_$vector --model ffnn --epoch 5000 --learn 0.00002 --hidden 3 --result ffnn_h3_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log ffnn_h4_$vector --model ffnn --epoch 5000 --learn 0.00002 --hidden 4 --result ffnn_h4_$vector --delete 1

	./run.sh --vector modeling/vectors/$vector --log cnn_h1_$vector --model cnn --epoch 5000 --learn 0.00002 --hidden 1 --result cnn_h1_$vector --delete 1
    ./run.sh --vector modeling/vectors/$vector --log cnn_h2_$vector --model cnn --epoch 5000 --learn 0.00002 --hidden 2 --result cnn_h2_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log cnn_h3_$vector --model cnn --epoch 5000 --learn 0.00002 --hidden 3 --result cnn_h3_$vector --delete 1
    ./run.sh --vector modeling/vectors/$vector --log cnn_h4_$vector --model cnn --epoch 5000 --learn 0.00002 --hidden 4 --result cnn_h4_$vector --delete 1
done