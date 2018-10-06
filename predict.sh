
echo
echo "sepsis"
echo

python predict.py -dir sepsis_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/sepsis_1 -closed 1 > result/sepsis_closed_1

python predict.py -dir sepsis_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/sepsis_except_1 -closed 1 > result/sepsis_except_closed_1


echo
echo "bacteremia"
echo

python predict.py -dir bacteremia_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/bacteremia_1 -closed 1 > result/bacteremia_closed_1

python predict.py -dir bacteremia_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/bacteremia_except_1 -closed 1 > result/bacteremia_except_closed_1

echo
echo "pneumonia"
echo

python predict.py -dir pneumonia_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/pneumonia_1 -closed 1 > result/pneumonia_closed_1

python predict.py -dir pneumonia_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/pneumonia_except_1 -closed 1 > result/pneumonia_except_closed_1


echo
echo "All"
echo

python predict.py -dir all_h_2_e_50000_l_000001 -epoch 5000 -hidden 2 -learn 0.000001 -show 1 -vector modeling/vectors/all_1 > result/all_closed_1

python predict.py -dir all_except_h_2_e_5000_l_000001 -epoch 5000 -hidden 2 -learn 0.000001 -show 1 -vector modeling/vectors/all_except_1 > result/all_except_closed_1


