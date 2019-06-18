#/bin/bash

for ((i=1;i<=10;i++))
do
    ./run.sh --model cnn --image_dir dataset/images/image_data/seoul/vector_36/ --vector modeling/vectors/seoul/vector --tensor_dir baseline_cnn_"$i" --hidden 3 --learn 0.00003 --epoch 500
done
