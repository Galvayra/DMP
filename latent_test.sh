learn=0.00001
lr='lr_00001'

learn2=0.00002
lr2='lr_00002'

learn3=0.00003
lr3='lr_00003'

learn4=0.00004
lr4='lr_00004'

learn5=0.00005
lr5='lr_00005'

v1='vector'
v2='vector_w2v'
v3='vector_feature_erase'

./run.sh --model cnn --vector modeling/vectors/"$v1" --image_dir dataset/images/image_data/"$v1"_50/ --log "$v1"_50_h_2_"$lr" --show 1 --learn "$learn" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v2" --image_dir dataset/images/image_data/"$v2"_50/ --log "$v2"_50_h_2_"$lr" --show 1 --learn "$learn" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v3" --image_dir dataset/images/image_data/"$v3"_50/ --log "$v3"_50_h_2_"$lr" --show 1 --learn "$learn" --hidden 2 --delete 1

./run.sh --model cnn --vector modeling/vectors/"$v1" --image_dir dataset/images/image_data/"$v1"_50/ --log "$v1"_50_h_2_"$lr2" --show 1 --learn "$learn2" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v2" --image_dir dataset/images/image_data/"$v2"_50/ --log "$v2"_50_h_2_"$lr2" --show 1 --learn "$learn2" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v3" --image_dir dataset/images/image_data/"$v3"_50/ --log "$v3"_50_h_2_"$lr2" --show 1 --learn "$learn2" --hidden 2 --delete 1

./run.sh --model cnn --vector modeling/vectors/"$v1" --image_dir dataset/images/image_data/"$v1"_50/ --log "$v1"_50_h_2_"$lr3" --show 1 --learn "$learn3" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v2" --image_dir dataset/images/image_data/"$v2"_50/ --log "$v2"_50_h_2_"$lr3" --show 1 --learn "$learn3" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v3" --image_dir dataset/images/image_data/"$v3"_50/ --log "$v3"_50_h_2_"$lr3" --show 1 --learn "$learn3" --hidden 2 --delete 1

./run.sh --model cnn --vector modeling/vectors/"$v1" --image_dir dataset/images/image_data/"$v1"_50/ --log "$v1"_50_h_2_"$lr4" --show 1 --learn "$learn4" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v2" --image_dir dataset/images/image_data/"$v2"_50/ --log "$v2"_50_h_2_"$lr4" --show 1 --learn "$learn4" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v3" --image_dir dataset/images/image_data/"$v3"_50/ --log "$v3"_50_h_2_"$lr4" --show 1 --learn "$learn4" --hidden 2 --delete 1

./run.sh --model cnn --vector modeling/vectors/"$v1" --image_dir dataset/images/image_data/"$v1"_50/ --log "$v1"_50_h_2_"$lr5" --show 1 --learn "$learn5" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v2" --image_dir dataset/images/image_data/"$v2"_50/ --log "$v2"_50_h_2_"$lr5" --show 1 --learn "$learn5" --hidden 2 --delete 1
./run.sh --model cnn --vector modeling/vectors/"$v3" --image_dir dataset/images/image_data/"$v3"_50/ --log "$v3"_50_h_2_"$lr5" --show 1 --learn "$learn5" --hidden 2 --delete 1

