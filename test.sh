#/bin/bash

./run.sh --vector modeling/vectors/all_sampling --dir ffnn_h_2 --model ffnn --epoch 7000 --learn 0.000005 --hidden 2 --show 1 --result ffnn_all_sampling_h_2
./run.sh --vector modeling/vectors/all_sampling --dir ffnn_h_3 --model ffnn --epoch 7000 --learn 0.000005 --hidden 3 --show 1 --result ffnn_all_sampling_h_3
./run.sh --vector modeling/vectors/all_sampling --dir ffnn_h_4 --model ffnn --epoch 7000 --learn 0.000005 --hidden 4 --show 1 --result ffnn_all_sampling_h_4
#./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_0 --model cnn --epoch 8000 --learn 0.000001 --hidden 0 --show 1 --result cnn_all_sampling_h_0
#./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_1 --model cnn --epoch 8000 --learn 0.000001 --hidden 1 --show 1 --result cnn_all_sampling_h_1
#./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_2 --model cnn --epoch 8000 --learn 0.000001 --hidden 2 --show 1 --result cnn_all_sampling_h_2
#./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_3 --model cnn --epoch 8000 --learn 0.000001 --hidden 3 --show 1 --result cnn_all_sampling_h_3
