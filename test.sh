
echo
echo "sepsis"
echo

python training.py -dir sepsis_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/sepsis_1 > result/sepsis_1

python training.py -dir sepsis_except_h_2_e_7000_l_00001 -epoch 7000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/sepsis_except_1 > result/sepsis_except_1


echo
echo "bacteremia"
echo

python training.py -dir bacteremia_h_2_e_10000_l_000005 -epoch 10000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/bacteremia_1 > result/bacteremia_1

python training.py -dir bacteremia_except_h_2_e_10000_l_000005 -epoch 10000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/bacteremia_except_1 > result/bacteremia_except_1

echo
echo "pneumonia"
echo

python training.py -dir pneumonia_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/pneumonia_1 > result/pneumonia_1

python training.py -dir pneumonia_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/pneumonia_except_1 > result/pneumonia_except_1


python training.py -dir pneumonia_h_2_e_10000_l_000005 -epoch 10000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/pneumonia_1 > result/pneumonia_2

python training.py -dir pneumonia_except_h_2_e_10000_l_000005 -epoch 10000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/pneumonia_except_1 > result/pneumonia_except_2

python training.py -dir pneumonia_except_h_2_e_15000_l_000005 -epoch 15000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/pneumonia_except_1 > result/pneumonia_except_3

echo
echo "All"
echo

python training.py -dir all_h_2_e_25000_l_01 -epoch 25000 -hidden 2 -learn 0.01 -show 1 -vector modeling/vectors/all_1 > result/all_1

python training.py -dir all_except_h_2_e_20000_l_01 -epoch 20000 -hidden 2 -learn 0.01 -show 1 -vector modeling/vectors/all_except_1 > result/all_except_1


