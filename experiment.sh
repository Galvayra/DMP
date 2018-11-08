#/bin/bash

## train ffnn
#./train.sh --vector modeling/vectors/all --log ffnn_h_1 --model ffnn --epoch 10000 --learn 0.000002 --hidden 1 --show 1 --result ffnn_all_h_1
#./train.sh --vector modeling/vectors/all --log ffnn_h_2 --model ffnn --epoch 10000 --learn 0.000002 --hidden 2 --show 1 --result ffnn_all_h_2
#./train.sh --vector modeling/vectors/all --log ffnn_h_3 --model ffnn --epoch 10000 --learn 0.000002 --hidden 3 --show 1 --result ffnn_all_h_3
#./train.sh --vector modeling/vectors/all --log ffnn_h_4 --model ffnn --epoch 10000 --learn 0.000002 --hidden 4 --show 1 --result ffnn_all_h_4

## train cnn
#./train.sh --vector modeling/vectors/all --log cnn_h_1 --model cnn --epoch 10000 --learn 0.000002 --hidden 1 --show 1 --result cnn_all_h_1
#./train.sh --vector modeling/vectors/all --log cnn_h_2 --model cnn --epoch 10000 --learn 0.000002 --hidden 2 --show 1 --result cnn_all_h_2
#./train.sh --vector modeling/vectors/all --log cnn_h_3 --model cnn --epoch 10000 --learn 0.000002 --hidden 3 --show 1 --result cnn_all_h_3
#./train.sh --vector modeling/vectors/all --log cnn_h_4 --model cnn --epoch 10000 --learn 0.000002 --hidden 4 --show 1 --result cnn_all_h_4


## test svm
#./test.sh --vector modeling/vectors/all --save svm --model svm
#
## test ffnn
#./test.sh --vector modeling/vectors/all --log ffnn_h_1 --model ffnn --epoch 10000
#./test.sh --vector modeling/vectors/all --log ffnn_h_2 --model ffnn --epoch 10000
#./test.sh --vector modeling/vectors/all --log ffnn_h_3 --model ffnn --epoch 10000
#./test.sh --vector modeling/vectors/all --log ffnn_h_4 --model ffnn --epoch 10000
#
## test cnn
#./test.sh --vector modeling/vectors/all --log cnn_h_1 --model cnn --epoch 10000
#./test.sh --vector modeling/vectors/all --log cnn_h_2 --model cnn --epoch 10000
#./test.sh --vector modeling/vectors/all --log cnn_h_3 --model cnn --epoch 10000
#./test.sh --vector modeling/vectors/all --log cnn_h_4 --model cnn --epoch 10000


./run.sh --vector modeling/vectors/all --log ffnn_1 --model ffnn --epoch 10000 --learn 0.000002 --hidden 1 --result ffnn_1 --delete 1
./run.sh --vector modeling/vectors/all --log ffnn_2 --model ffnn --epoch 10000 --learn 0.000002 --hidden 2 --result ffnn_2 --delete 1
./run.sh --vector modeling/vectors/all --log ffnn_3 --model ffnn --epoch 10000 --learn 0.000002 --hidden 3 --result ffnn_3 --delete 1
./run.sh --vector modeling/vectors/all --log ffnn_4 --model ffnn --epoch 10000 --learn 0.000002 --hidden 4 --result ffnn_4 --delete 1
./run.sh --vector modeling/vectors/all --log cnn_1 --model cnn --epoch 10000 --learn 0.000001 --hidden 1 --result cnn_1 --delete 1
./run.sh --vector modeling/vectors/all --log cnn_2 --model cnn --epoch 10000 --learn 0.000001 --hidden 2 --result cnn_2 --delete 1
./run.sh --vector modeling/vectors/all --log cnn_3 --model cnn --epoch 10000 --learn 0.000001 --hidden 3 --result cnn_3 --delete 1
./run.sh --vector modeling/vectors/all --log cnn_4 --model cnn --epoch 10000 --learn 0.000001 --hidden 4 --result cnn_4 --delete 1

./run.sh --vector modeling/vectors/all_w2v --log ffnn_w2v_1 --model ffnn --epoch 5000 --learn 0.000002 --hidden 1 --result ffnn_w2v_1 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log ffnn_w2v_2 --model ffnn --epoch 5000 --learn 0.000002 --hidden 2 --result ffnn_w2v_2 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log ffnn_w2v_3 --model ffnn --epoch 5000 --learn 0.000002 --hidden 3 --result ffnn_w2v_3 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log ffnn_w2v_4 --model ffnn --epoch 5000 --learn 0.000002 --hidden 4 --result ffnn_w2v_4 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log cnn_w2v_1 --model cnn --epoch 5000 --learn 0.000002 --hidden 1 --result cnn_w2v_1 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log cnn_w2v_2 --model cnn --epoch 5000 --learn 0.000002 --hidden 2 --result cnn_w2v_2 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log cnn_w2v_3 --model cnn --epoch 5000 --learn 0.000002 --hidden 3 --result cnn_w2v_3 --delete 1
./run.sh --vector modeling/vectors/all_w2v --log cnn_w2v_4 --model cnn --epoch 5000 --learn 0.000002 --hidden 4 --result cnn_w2v_4 --delete 1

./run.sh --vector modeling/vectors/all_w2v_scaling --log ffnn_w2v_scaling_1 --model ffnn --epoch 10000 --learn 0.000002 --hidden 1 --result ffnn_w2v_scaling_1 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log ffnn_w2v_scaling_2 --model ffnn --epoch 10000 --learn 0.000002 --hidden 2 --result ffnn_w2v_scaling_2 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log ffnn_w2v_scaling_3 --model ffnn --epoch 10000 --learn 0.000002 --hidden 3 --result ffnn_w2v_scaling_3 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log ffnn_w2v_scaling_4 --model ffnn --epoch 10000 --learn 0.000002 --hidden 4 --result ffnn_w2v_scaling_4 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log cnn_w2v_scaling_1 --model cnn --epoch 10000 --learn 0.000001 --hidden 1 --result cnn_w2v_scaling_1 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log cnn_w2v_scaling_2 --model cnn --epoch 10000 --learn 0.000001 --hidden 2 --result cnn_w2v_scaling_2 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log cnn_w2v_scaling_3 --model cnn --epoch 10000 --learn 0.000001 --hidden 3 --result cnn_w2v_scaling_3 --delete 1
./run.sh --vector modeling/vectors/all_w2v_scaling --log cnn_w2v_scaling_4 --model cnn --epoch 10000 --learn 0.000001 --hidden 4 --result cnn_w2v_scaling_4 --delete 1