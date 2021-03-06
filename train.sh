#/bin/bash

epoch=10000
learn=0.00001
hidden=2
show=1
delete=0
tensor_dir=""
model=ffnn
vector=""
result=""
feature=""
target=""
image_dir=""
ver=1

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

if [ "$vector" == "" ]; then
    echo
	echo "please input vector"
	echo
else
    if [ "$result" == "" ]; then
        python training.py -vector "$vector" -tensor_dir "$tensor_dir" -model "$model" -show "$show" -delete "$delete" -epoch "$epoch" -hidden "$hidden" -learn "$learn" -feature "$feature" -target "$target" -image_dir "$image_dir" -ver "$ver"
    else
        python training.py -vector "$vector" -tensor_dir "$tensor_dir" -model "$model" -show "$show" -delete "$delete" -epoch "$epoch" -hidden "$hidden" -learn "$learn" -feature "$feature" -target "$target" -image_dir "$image_dir" -ver "$ver" > result/"$result"
    fi
fi
