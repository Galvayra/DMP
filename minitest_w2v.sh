## train ffnn
#/bin/bash

# w2v를 사용한 경우 loss 감소율이 2배로 늦음... (scaling 유무 상관 x)
# w2v를 사용하지 않은 경우 FFNN --> 5000 epoch // lr 4e-06  // cnn --> 10000 epoch // lr 2e-06
# 좀 더 epoch을 줄이고 lr을 올릴 필요가 있음


# minimal test

for vector in $@
do
	echo "test" $vector

    # if not w2v lr 4e-06 && 5000 epoch
#	./run.sh --vector modeling/vectors/$vector --log ffnn_h1_$vector --model ffnn --epoch 4000 --learn 0.00002 --hidden 1 --result ffnn_h1_$vector --delete 1
#	./run.sh --vector modeling/vectors/$vector --log ffnn_h2_$vector --model ffnn --epoch 4000 --learn 0.00001 --hidden 2 --result ffnn_h2_$vector --delete 1
#	./run.sh --vector modeling/vectors/$vector --log ffnn_h3_$vector --model ffnn --epoch 4000 --learn 0.00001 --hidden 3 --result ffnn_h3_$vector --delete 1
#	./run.sh --vector modeling/vectors/$vector --log ffnn_h4_$vector --model ffnn --epoch 4000 --learn 0.00002 --hidden 4 --result ffnn_h4_$vector --delete 1

#    # if not  w2v lr 2e-06 && 10000 epoch
#    #         w2v lr 2e-05 && 3000 epoch
	./run.sh --vector modeling/vectors/$vector --log cnn_h1_$vector --model cnn --epoch 4000 --learn 0.00001 --hidden 1 --result cnn_h1_$vector --delete 1
    ./run.sh --vector modeling/vectors/$vector --log cnn_h2_$vector --model cnn --epoch 4000 --learn 0.00001 --hidden 2 --result cnn_h2_$vector --delete 1
	./run.sh --vector modeling/vectors/$vector --log cnn_h3_$vector --model cnn --epoch 4000 --learn 0.00001 --hidden 3 --result cnn_h3_$vector --delete 1
    ./run.sh --vector modeling/vectors/$vector --log cnn_h4_$vector --model cnn --epoch 4000 --learn 0.00001 --hidden 4 --result cnn_h4_$vector --delete 1
done