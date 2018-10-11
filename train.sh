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

echo
echo "seoul sampling except"
echo

python training.py -dir seoul_sampling_except_h_2_e_7000_l_00001 -epoch 7000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling_except_5 > result/seoul_sampling_except_2

python training.py -dir seoul_sampling_except_all_h_2_e_7000_l_00001 -epoch 7000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling_except_all_5 > result/seoul_sampling_except_all_2

echo
echo "seoul sampling final diagnosis only class"
echo

python training.py -dir seoul_sampling_h_2_e_7000_l_00001 -epoch 7000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling_5 > result/seoul_sampling_2

python training.py -dir seoul_sampling_all_h_2_e_7000_l_00001 -epoch 7000 -hidden 2 -learn 0.00001 -show 1 -vector modeling/vectors/seoul_sampling_all_5 > result/seoul_sampling_all_2

echo
echo "seoul sampling all of final diagnosis"
echo

python training.py -dir seoul_sampling_diagnosis_h_2_e_10000_l_000005 -epoch 10000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling_diagnosis_5 > result/seoul_sampling_diagnosis_1

python training.py -dir seoul_sampling_diagnosis_all_h_2_e_10000_l_000005 -epoch 10000 -hidden 2 -learn 0.000005 -show 1 -vector modeling/vectors/seoul_sampling_diagnosis_all_5 > result/seoul_sampling_diagnosis_all_1
