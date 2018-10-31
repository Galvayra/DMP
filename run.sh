#/bin/bash

./run.sh --vector modeling/vectors/all_sampling --log ffnn_h_1 --model ffnn --epoch 10000 --learn 0.000002 --hidden 1 --show 1 --result ffnn_all_sampling_h_1
./run.sh --vector modeling/vectors/all_sampling --log ffnn_h_2 --model ffnn --epoch 10000 --learn 0.000002 --hidden 2 --show 1 --result ffnn_all_sampling_h_2
./run.sh --vector modeling/vectors/all_sampling --log ffnn_h_3 --model ffnn --epoch 10000 --learn 0.000002 --hidden 3 --show 1 --result ffnn_all_sampling_h_3
./run.sh --vector modeling/vectors/all_sampling --log ffnn_h_4 --model ffnn --epoch 10000 --learn 0.000002 --hidden 4 --show 1 --result ffnn_all_sampling_h_4

./run.sh --vector modeling/vectors/all_sampling --log cnn_h_1 --model cnn --epoch 10000 --learn 0.000001 --hidden 1 --show 1 --result cnn_all_sampling_h_1
./run.sh --vector modeling/vectors/all_sampling --log cnn_h_2 --model cnn --epoch 10000 --learn 0.000001 --hidden 2 --show 1 --result cnn_all_sampling_h_2
./run.sh --vector modeling/vectors/all_sampling --log cnn_h_3 --model cnn --epoch 10000 --learn 0.000001 --hidden 3 --show 1 --result cnn_all_sampling_h_3
./run.sh --vector modeling/vectors/all_sampling --log cnn_h_4 --model cnn --epoch 10000 --learn 0.000001 --hidden 4 --show 1 --result cnn_all_sampling_h_4
