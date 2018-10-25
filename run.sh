#/bin/bash

epoch=10000
learn=0.00001
hidden=2
show=1
plot=0
dir=""
model=ffnn
vector=""
result=""

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

if [ "$vector" == "" ]; then
    echo
	echo "please input vector"
	echo
else
	python training.py -vector "$vector"
fi
