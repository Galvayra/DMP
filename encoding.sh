#/bin/sh

base=""
input=""
intermediate=""
output=""
softmax=0
sampling=0
parsing=0
encode_image=0

#PYTHONPATH=$PYTHONPATH:~/Project
#export PYTHONPATH

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

if [ $parsing -eq 1 ];then
    python parsing.py -input "$input" -output "$intermediate" -sampling "$sampling" -parsing_image 1
    echo
fi
echo
python encoding.py -base "$base" -input "$intermediate" -output "$output" -w2v 0 -ver 1 -softmax "$softmax" -encode_image "$encode_image"
echo
