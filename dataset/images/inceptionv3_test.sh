echo "vector_scale"
echo 
echo

python inceptionv3_retrain.py --image_dir dataset/vector_scale/ --log_path log/vector --save_path batch_256_epoch_2000_lr_01 --train_batch_size 256 --epoch 2000 --validation_batch_size 256 --save_step_interval 200 --eval_step_interval 200


echo "vector"
echo
echo

python inceptionv3_retrain.py --image_dir dataset/vector/ --log_path log/vector --save_path batch_256_epoch_2000_lr_01 --train_batch_size 256 --epoch 2000 --validation_batch_size 256 --save_step_interval 200 --eval_step_interval 200


echo "vector_feature_erase_scale"
echo 
echo

python inceptionv3_retrain.py --image_dir dataset/vector_feature_erase_scale/ --log_path log/vector --save_path batch_256_epoch_2000_lr_01 --train_batch_size 256 --epoch 2000 --validation_batch_size 256 --save_step_interval 200 --eval_step_interval 200


echo "vector_feature_erase"
echo 
echo

python inceptionv3_retrain.py --image_dir dataset/vector_feature_erase/ --log_path log/vector --save_path batch_256_epoch_2000_lr_01 --train_batch_size 256 --epoch 2000 --validation_batch_size 256 --save_step_interval 200 --eval_step_interval 200


echo "vector_w2v_scale"
echo 
echo

python inceptionv3_retrain.py --image_dir dataset/vector_w2v_scale/ --log_path log/vector --save_path batch_256_epoch_2000_lr_01 --train_batch_size 256 --epoch 2000 --validation_batch_size 256 --save_step_interval 200 --eval_step_interval 200

echo "vector_w2v"
echo 
echo

python inceptionv3_retrain.py --image_dir dataset/vector_w2v/ --log_path log/vector --save_path batch_256_epoch_2000_lr_01 --train_batch_size 256 --epoch 2000 --validation_batch_size 256 --save_step_interval 200 --eval_step_interval 200


