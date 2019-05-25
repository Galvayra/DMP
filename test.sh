#/bin/bash

learn=0.00001
hidden=2
show=0
plot=0
log=""
model=ffnn
vector=""
save=""
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
    if [ "$save" == "" ]; then
        python predict.py -vector "$vector" -log "$log" -model "$model" -show "$show" -save "$log" -plot "$plot" -feature "$feature" -target "$target" -image_dir "$image_dir" -ver "$ver"
    else
        python predict.py -vector "$vector" -log "$log" -model "$model" -show "$show" -save "$save" -plot "$plot" -feature "$feature" -target "$target" -image_dir "$image_dir" -ver "$ver"
    fi
fi
