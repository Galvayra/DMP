#/bin/bash

epoch=4000
learn=0.00001
hidden=2
show=1
delete=1
tensor_dir=""
model=ffnn
vector=""
plot=0
feature=""
target=""
image_dir=""
ver=1
#save=""
#result=""

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

echo
echo "################# Training !! #################"
echo
./train.sh --vector "$vector" --model "$model" --show "$show" --tensor_dir "$tensor_dir" --delete "$delete" --epoch "$epoch" --hidden "$hidden" --learn "$learn" --feature "$feature" --target "$target" --image_dir "$image_dir" --ver "$ver"

echo
echo "################# Test !! #################"
echo
./test.sh --vector "$vector" --model "$model" --show "$show" --tensor_dir "$tensor_dir" --plot "$plot" --save "$tensor_dir" --feature "$feature" --target "$target" --image_dir "$image_dir" --ver "$ver"
