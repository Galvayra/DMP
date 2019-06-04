#/bin/bash


#for ((i=1;i<=6;i++))
#do
#./run.sh --model ffnn --vector modeling/vectors/set_2/vector_standard --epoch 400 --learn 0.00005 --show 1 --tensor_dir set_2_standard_h_"$i" --delete 1 --hidden "$i"
#done

#./run.sh --model ffnn --vector modeling/vectors/set_2/vector --epoch 400 --learn 0.00002 --show 1 --tensor_dir set_2_vector --delete 1 --hidden 2
#./run.sh --model ffnn --vector modeling/vectors/set_2/vector --epoch 400 --learn 0.00002 --show 1 --tensor_dir set_2_vector --delete 1 --hidden 3

#./run.sh --model ffnn --vector modeling/vectors/set_2/vector_all --epoch 400 --learn 0.00002 --show 1 --tensor_dir set_2_vector_all --delete 1 --hidden 2
#./run.sh --model ffnn --vector modeling/vectors/set_2/vector_all --epoch 400 --learn 0.00002 --show 1 --tensor_dir set_2_vector_all --delete 1 --hidden 3

#./run.sh --model ffnn --vector modeling/vectors/set_2/vector_all --epoch 400 --learn 0.00002 --show 1 --tensor_dir set_2_vector_all --delete 1 --hidden 3


#for ((i=2;i<=9;i++))
#do
#python predict.py -model svm -show 1 -vector modeling/vectors/set_"$i"/vector -save set_"$i"_svm
#done
#

for ((i=2;i<=9;i++))
do
./run.sh --model ffnn --vector modeling/vectors/set_"$i"/vector --epoch 1000 --learn 0.00001 --show 1 --tensor_dir set_"$i"_ffnn_h2 --delete 1 --hidden 2
done


for ((i=2;i<=9;i++))
do
./run.sh --model ffnn --vector modeling/vectors/set_"$i"/vector --epoch 1000 --learn 0.00001 --show 1 --tensor_dir set_"$i"_ffnn_h3 --delete 1 --hidden 3
done


for ((i=2;i<=9;i++))
do
./run.sh --model ffnn --vector modeling/vectors/set_"$i"/vector --epoch 1000 --learn 0.00001 --show 1 --tensor_dir set_"$i"_ffnn_h4 --delete 1 --hidden 4
done


for ((i=2;i<=9;i++))
do
./run.sh --model ffnn --vector modeling/vectors/set_"$i"/vector --epoch 1000 --learn 0.00001 --show 1 --tensor_dir set_"$i"_ffnn_h5 --delete 1 --hidden 5
done


for ((i=2;i<=9;i++))
do
./run.sh --model ffnn --vector modeling/vectors/set_"$i"/vector --epoch 1000 --learn 0.00001 --show 1 --tensor_dir set_"$i"_ffnn_h6 --delete 1 --hidden 6
done