#/bin/bash

./run.sh --vector modeling/vectors/all_sampling --dir ffnn_h_2_e_6000_lr_00001 --model ffnn --epoch 6000 --learn 0.000001 --hidden 2 --show 1 --result ffnn_all_sampling_h_2
./run.sh --vector modeling/vectors/all_sampling --dir ffnn_h_3_e_6000_lr_00001 --model ffnn --epoch 6000 --learn 0.000001 --hidden 3 --show 1 --result ffnn_all_sampling_h_3
./run.sh --vector modeling/vectors/all_sampling --dir ffnn_h_4_e_6000_lr_00001 --model ffnn --epoch 6000 --learn 0.000001 --hidden 4 --show 1 --result ffnn_all_sampling_h_4
./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_0_e_6000_lr_00001 --model cnn --epoch 6000 --learn 0.000001 --hidden 0 --show 1 --result cnn_all_sampling_h_0
./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_1_e_6000_lr_00001 --model cnn --epoch 6000 --learn 0.000001 --hidden 1 --show 1 --result cnn_all_sampling_h_1
./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_2_e_6000_lr_00001 --model cnn --epoch 6000 --learn 0.000001 --hidden 2 --show 1 --result cnn_all_sampling_h_2
./run.sh --vector modeling/vectors/all_sampling --dir cnn_h_3_e_6000_lr_00001 --model cnn --epoch 6000 --learn 0.000001 --hidden 3 --show 1 --result cnn_all_sampling_h_3
