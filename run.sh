#/bin/bash

epoch=2000
learn=0.00001
hidden=2
show=1
delete=0
log=""
model=ffnn
vector=""
plot=0
feature=""
target=""
image_dir=""
#save=""
#result=""

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

echo
echo "################# Training !! #################"
echo
./train.sh --vector "$vector" --model "$model" --show "$show" --log "$log" --delete "$delete" --result "$result" --epoch "$epoch" --hidden "$hidden" --learn "$learn" --feature "$feature" --target "$target" --image_dir "$image_dir"

echo
echo "################# Test !! #################"
echo
./test.sh --vector "$vector" --model "$model" --show "$show" --log "$log" --plot "$plot" --save "$log" --feature "$feature" --target "$target" --image_dir "$image_dir"
