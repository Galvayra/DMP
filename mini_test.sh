#/bin/bash

#for ((i=1;i<=5;i++))
#do
#    ./run.sh --model ffnn --vector modeling/vectors/vector --tensor_dir vector_ffnn_h3_"$i" --hidden 3 --learn 0.00001 --epoch 500
#    ./run.sh --model ffnn --vector modeling/vectors/vector --tensor_dir vector_ffnn_h4_"$i" --hidden 4 --learn 0.00001 --epoch 500
#    ./run.sh --model ffnn --vector modeling/vectors/vector --tensor_dir vector_ffnn_h5_"$i" --hidden 5 --learn 0.00001 --epoch 500
#    ./run.sh --model ffnn --vector modeling/vectors/vector_all --tensor_dir vector_all_ffnn_h3_"$i" --hidden 3 --learn 0.00001 --epoch 500
#    ./run.sh --model ffnn --vector modeling/vectors/vector_all --tensor_dir vector_all_ffnn_h4_"$i" --hidden 4 --learn 0.00001 --epoch 500
#    ./run.sh --model ffnn --vector modeling/vectors/vector_all --tensor_dir vector_all_ffnn_h5_"$i" --hidden 5 --learn 0.00001 --epoch 500
#    ./run.sh --model cnn --image_dir dataset/images/image_data/vector/ --vector modeling/vectors/vector --tensor_dir vector_cnn_"$i" --hidden 3 --learn 0.00001 --epoch 500
#    ./run.sh --model cnn --image_dir dataset/images/image_data/vector_all/ --vector modeling/vectors/vector_all --tensor_dir vector_all_cnn_"$i" --hidden 3 --learn 0.00001 --epoch 500
#done


#./run.sh --model ffnn --vector modeling/vectors/vector_1 --tensor_dir vector_1 --hidden 5 --learn 0.00002 --epoch 500
#./run.sh --model ffnn --vector modeling/vectors/vector_2 --tensor_dir vector_2 --hidden 5 --learn 0.00002 --epoch 500
#./run.sh --model ffnn --vector modeling/vectors/vector_3 --tensor_dir vector_3 --hidden 5 --learn 0.00002 --epoch 500
#./run.sh --model ffnn --vector modeling/vectors/vector_4 --tensor_dir vector_4 --hidden 5 --learn 0.00002 --epoch 500
#./run.sh --model ffnn --vector modeling/vectors/vector_5 --tensor_dir vector_5 --hidden 5 --learn 0.00002 --epoch 500



./run.sh --model ffnn --vector modeling/vectors/vector_1_minmax --tensor_dir vector_1_minmax --hidden 5 --learn 0.00001 --epoch 500
./run.sh --model ffnn --vector modeling/vectors/vector_2_minmax --tensor_dir vector_2_minmax --hidden 5 --learn 0.00001 --epoch 500
./run.sh --model ffnn --vector modeling/vectors/vector_3_minmax --tensor_dir vector_3_minmax --hidden 5 --learn 0.00002 --epoch 500
./run.sh --model ffnn --vector modeling/vectors/vector_4_minmax --tensor_dir vector_4_minmax --hidden 5 --learn 0.00002 --epoch 500
./run.sh --model ffnn --vector modeling/vectors/vector_5_minmax --tensor_dir vector_5_minmax --hidden 5 --learn 0.00002 --epoch 500
