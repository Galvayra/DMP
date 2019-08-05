#/bin/sh

base=""
input="dataset_images"
output="image_vector"
softmax=0
parsing=0

PYTHONPATH=$PYTHONPATH:~/Project
export PYTHONPATH

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

if [ $parsing -eq 1 ];then
    python parsing.py -output "$input" -sampling 0 -ratio 0.6 -parsing_image 1
    echo
fi
echo
python encoding.py -base "$base" -input "$input" -output "$output" -w2v 0 -ver 1 -softmax "$softmax" -encode_image 1
echo
