#/bin/bash

image_dir=dataset/images/image_data/vector_paper/
vector=modeling/vectors/vector_paper
model=ffnn
name=paper
learn=00001z
log="$name"_"$model"_lr

for n in {1..50}
do
    ./run.sh --vector "$vector" --log "$log"_00002_"$n" --learn 0.00002 --hidden 4 --model "$model"
done
