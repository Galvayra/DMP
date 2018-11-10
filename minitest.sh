## train ffnn
#/bin/bash


# minimal test 


for vector in $@
do
	echo "test" $vector
	./run.sh --vector modeling/vectors/$vector --log ffnn_3_$vector --model ffnn --epoch 5000 --learn 0.000004 --hidden 3 --result ffnn_3_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log ffnn_4_$vector --model ffnn --epoch 5000 --learn 0.000004 --hidden 4 --result ffnn_4_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log cnn_3_$vector --model cnn --epoch 5000 --learn 0.000005 --hidden 3 --result cnn_3_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log cnn_4_$vector --model cnn --epoch 5000 --learn 0.000005 --hidden 4 --result cnn_4_$vector --delete 1
done
#./run.sh --vector modeling/vectors/all_w2v --log ffnn_w2v_3 --model ffnn --epoch 4000 --learn 0.00001 --hidden 3 --result ffnn_w2v_3 --delete 1
#./run.sh --vector modeling/vectors/all_w2v_scaling --log ffnn_w2v_scaling_3 --model ffnn --epoch 4000 --learn 0.00001 --hidden 3 --result ffnn_w2v_scaling_4 --delete 1
#./run.sh --vector modeling/vectors/all_w2v_scaling --log cnn_w2v_scaling_3 --model cnn --epoch 5000 --learn 0.00001 --hidden 3 --result cnn_w2v_scaling_3 --delete 1
