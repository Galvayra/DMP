#/bin/bash

epoch=10000
learn=0.00001
hidden=2
show=1
delete=0
log=""
model=ffnn
vector=""
result=""
plot=0
save=""
feature=""

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

echo
echo "################# Training !! #################"
echo
./train.sh --vector "$vector" --model "$model" --show "$show" --log "$log" --delete "$delete" --result "$result" --epoch "$epoch" --hidden "$hidden" --learn "$learn" --feature "$feature"

echo
echo "################# Test !! #################"
echo
./test.sh --vector "$vector" --model "$model" --show "$show" --log "$log" --plot "$plot" --epoch "$epoch" --save "$save" --feature "$feature"