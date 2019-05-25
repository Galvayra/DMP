#/bin/bash

epoch=4000
learn=0.00001
hidden=2
show=1
delete=1
log=""
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
./train.sh --vector "$vector" --model "$model" --show "$show" --log "$log" --delete "$delete" --epoch "$epoch" --hidden "$hidden" --learn "$learn" --feature "$feature" --target "$target" --image_dir "$image_dir" --ver "$ver"

echo
echo "################# Test !! #################"
echo
./test.sh --vector "$vector" --model "$model" --show "$show" --log "$log" --plot "$plot" --save "$log" --feature "$feature" --target "$target" --image_dir "$image_dir" --ver "$ver"
