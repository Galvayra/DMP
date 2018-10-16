#
#echo
#echo "sepsis"
#echo
#
#python training.py -dir sepsis_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/sepsis_1 > result/sepsis_1
#
#python training.py -dir sepsis_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/sepsis_except_1 > result/sepsis_except_1
#
#
#echo
#echo "bacteremia"
#echo
#
#python training.py -dir bacteremia_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/bacteremia_1 > result/bacteremia_1
#
#python training.py -dir bacteremia_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/bacteremia_except_1 > result/bacteremia_except_1
#
#echo
#echo "pneumonia"
#echo
#
#python training.py -dir pneumonia_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/pneumonia_1 > result/pneumonia_1
#
#python training.py -dir pneumonia_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/pneumonia_except_1 > result/pneumonia_except_1
#
#
#echo
#echo "All"
#echo
#
#python training.py -dir all_h_2_e_5000_l_000005 -epoch 5000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/all_1 > result/all_1
#
#python training.py -dir all_except_h_2_e_5000_l_000005 -epoch 5000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/all_except_1 > result/all_except_1
#
#

#echo
#echo "seoul sampling except"
#echo
#
#python training.py -dir seoul_sampling2_except_h_1_e_5000_l_00001 -epoch 5000 -hidden 1 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_5 > result/seoul_sampling2_except_1
#
#python training.py -dir seoul_sampling2_except_all_h_1_e_5000_l_00001 -epoch 5000 -hidden 1 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_all_5 > result/seoul_sampling2_except_all_1
#
#python training.py -dir seoul_sampling2_except_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_5 > result/seoul_sampling2_except_2
#
#python training.py -dir seoul_sampling2_except_all_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_all_5 > result/seoul_sampling2_except_all_2
#
#python training.py -dir seoul_sampling2_except_h_3_e_5000_l_00001 -epoch 5000 -hidden 3 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_5 > result/seoul_sampling2_except_3
#
#python training.py -dir seoul_sampling2_except_all_h_3_e_5000_l_00001 -epoch 5000 -hidden 3 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_all_5 > result/seoul_sampling2_except_all_3
#
#python training.py -dir seoul_sampling2_except_h_4_e_5000_l_00001 -epoch 5000 -hidden 4 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_5 > result/seoul_sampling2_except_4
#
#python training.py -dir seoul_sampling2_except_all_h_4_e_5000_l_00001 -epoch 5000 -hidden 4 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_except_all_5 > result/seoul_sampling2_except_all_4

echo
echo "seoul sampling final diagnosis only class"
echo

python training.py -dir seoul_sampling2_h_1_e_5000_l_00001 -epoch 5000 -hidden 1 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_5 > result/seoul_sampling2_1

python training.py -dir seoul_sampling2_all_h_1_e_5000_l_00001 -epoch 5000 -hidden 1 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_all_5 > result/seoul_sampling2_all_1

python training.py -dir seoul_sampling2_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_5 > result/seoul_sampling2_2

python training.py -dir seoul_sampling2_all_h_2_e_5000_l_00001 -epoch 5000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_all_5 > result/seoul_sampling2_all_2

python training.py -dir seoul_sampling2_h_3_e_5000_l_00001 -epoch 5000 -hidden 3 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_5 > result/seoul_sampling2_3

python training.py -dir seoul_sampling2_all_h_3_e_5000_l_00001 -epoch 5000 -hidden 3 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_all_5 > result/seoul_sampling2_all_3

python training.py -dir seoul_sampling2_h_4_e_5000_l_00001 -epoch 5000 -hidden 4 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_5 > result/seoul_sampling2_4

python training.py -dir seoul_sampling2_all_h_4_e_5000_l_00001 -epoch 5000 -hidden 4 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling2_all_5 > result/seoul_sampling2_all_4


echo
echo "seoul sampling all of final diagnosis"
echo

python training.py -dir seoul_sampling2_diagnosis_h_1_e_6000_l_000005 -epoch 6000 -hidden 1 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_5 > result/seoul_sampling2_diagnosis_1

python training.py -dir seoul_sampling2_diagnosis_all_h_1_e_6000_l_000005 -epoch 6000 -hidden 1 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_all_5 > result/seoul_sampling2_diagnosis_all_1

python training.py -dir seoul_sampling2_diagnosis_h_2_e_6000_l_000005 -epoch 6000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_5 > result/seoul_sampling2_diagnosis_2

python training.py -dir seoul_sampling2_diagnosis_all_h_2_e_6000_l_000005 -epoch 6000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_all_5 > result/seoul_sampling2_diagnosis_all_2

python training.py -dir seoul_sampling2_diagnosis_h_3_e_6000_l_000005 -epoch 6000 -hidden 3 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_5 > result/seoul_sampling2_diagnosis_3

python training.py -dir seoul_sampling2_diagnosis_all_h_3_e_6000_l_000005 -epoch 6000 -hidden 3 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_all_5 > result/seoul_sampling2_diagnosis_all_3

python training.py -dir seoul_sampling2_diagnosis_h_4_e_6000_l_000005 -epoch 6000 -hidden 4 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_5 > result/seoul_sampling2_diagnosis_4

python training.py -dir seoul_sampling2_diagnosis_all_h_4_e_6000_l_000005 -epoch 6000 -hidden 4 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling2_diagnosis_all_5 > result/seoul_sampling2_diagnosis_all_4
