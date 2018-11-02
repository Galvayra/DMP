#/bin/bash

## train ffnn
#./train.sh --vector modeling/vectors/all_sampling --log ffnn_h_1 --model ffnn --epoch 10000 --learn 0.000002 --hidden 1 --show 1 --result ffnn_all_sampling_h_1
#./train.sh --vector modeling/vectors/all_sampling --log ffnn_h_2 --model ffnn --epoch 10000 --learn 0.000002 --hidden 2 --show 1 --result ffnn_all_sampling_h_2
#./train.sh --vector modeling/vectors/all_sampling --log ffnn_h_3 --model ffnn --epoch 10000 --learn 0.000002 --hidden 3 --show 1 --result ffnn_all_sampling_h_3
#./train.sh --vector modeling/vectors/all_sampling --log ffnn_h_4 --model ffnn --epoch 10000 --learn 0.000002 --hidden 4 --show 1 --result ffnn_all_sampling_h_4

## train cnn
#./train.sh --vector modeling/vectors/all_sampling --log cnn_h_1 --model cnn --epoch 10000 --learn 0.000001 --hidden 1 --show 1 --result cnn_all_sampling_h_1
#./train.sh --vector modeling/vectors/all_sampling --log cnn_h_2 --model cnn --epoch 10000 --learn 0.000001 --hidden 2 --show 1 --result cnn_all_sampling_h_2
#./train.sh --vector modeling/vectors/all_sampling --log cnn_h_3 --model cnn --epoch 10000 --learn 0.000001 --hidden 3 --show 1 --result cnn_all_sampling_h_3
#./train.sh --vector modeling/vectors/all_sampling --log cnn_h_4 --model cnn --epoch 10000 --learn 0.000001 --hidden 4 --show 1 --result cnn_all_sampling_h_4


# test svm
./test.sh --vector modeling/vectors/all_sampling --save svm --model svm

# test ffnn
./test.sh --vector modeling/vectors/all_sampling --log ffnn_h_1 --model ffnn --epoch 10000
./test.sh --vector modeling/vectors/all_sampling --log ffnn_h_2 --model ffnn --epoch 10000
./test.sh --vector modeling/vectors/all_sampling --log ffnn_h_3 --model ffnn --epoch 10000
./test.sh --vector modeling/vectors/all_sampling --log ffnn_h_4 --model ffnn --epoch 10000

# test cnn
./test.sh --vector modeling/vectors/all_sampling --log cnn_h_1 --model cnn --epoch 10000
./test.sh --vector modeling/vectors/all_sampling --log cnn_h_2 --model cnn --epoch 10000
./test.sh --vector modeling/vectors/all_sampling --log cnn_h_3 --model cnn --epoch 10000
./test.sh --vector modeling/vectors/all_sampling --log cnn_h_4 --model cnn --epoch 10000
