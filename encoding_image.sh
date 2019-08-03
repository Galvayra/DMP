#/bin/sh

base=""
input="image"
output="image_vector"
softmax=0
parsing=0

PYTHONPATH=$PYTHONPATH:~/Project
export PYTHONPATH

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

if [ $parsing -eq 1 ];then
    python dataset/images/parsing.py -output "$input" -sampling 0 -ratio 0.6
    echo
fi
echo
python dataset/images/encoding.py -base "$base" -input "$input" -output "$output" -w2v 0 -ver 1 -softmax "$softmax"
echo
