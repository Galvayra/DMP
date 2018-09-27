
echo
echo "sepsis"
echo

python training.py -dir sepsis_except_h_2_e_10000_l_008 -epoch 10000 -hidden 2 -learn 0.008 -show 0 -vector modeling/vectors/sepsis_except_fd_1 > result/sepsis_except

echo
echo "bacteremia"
echo

python training.py -dir bacteremia_except_h_2_e_10000_l_008 -epoch 10000 -hidden 2 -learn 0.008 -show 0 -vector modeling/vectors/bacteremia_except_fd_1 > result/bacteremia_except

echo
echo "pneumonia"
echo

python training.py -dir pneumonia_except_h_2_e_8000_l_008 -epoch 8000 -hidden 2 -learn 0.008 -show 0 -vector modeling/vectors/pneumonia_except_fd_1 > result/pneumonia_except

echo
echo "All"
echo

python training.py -dir all_h_2_e_20000_l_008 -epoch 20000 -hidden 2 -learn 0.008 -show 0 -vector modeling/vectors/all_1 > result/all

echo
echo "All except"
echo

python training.py -dir all_except_h_2_e_20000_l_008 -epoch 20000 -hidden 2 -learn 0.008 -show 0 -vector modeling/vectors/all_except_fd_1 > result/all_except

