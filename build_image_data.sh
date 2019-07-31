#/bin/sh

input="image"
output="image_vector"
softmax=1

PYTHONPATH=$PYTHONPATH:~/Project
export PYTHONPATH

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

python dataset/images/parsing.py -output $input -sampling 0 -ratio 0.6
echo
echo
python dataset/images/encoding.py -input $input -output $output -w2v 0 -ver 1 -softmax $softmax
echo
