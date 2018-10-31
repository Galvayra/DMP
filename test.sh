#/bin/bash

epoch=10000
learn=0.00001
hidden=2
show=0
plot=0
log=""
model=ffnn
vector=""
save=""

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

if [ "$vector" == "" ]; then
    echo
	echo "please input vector"
	echo
else
    if [ "$save" == "" ]; then
        python predict.py -vector "$vector" -log "$log" -model "$model" -show "$show" -epoch "$epoch" -save "$log" -plot "$plot"
    else
        python predict.py -vector "$vector" -log "$log" -model "$model" -show "$show" -epoch "$epoch" -save "$save" -plot "$plot"
    fi
fi
