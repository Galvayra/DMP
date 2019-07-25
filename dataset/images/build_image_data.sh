#!/bin/sh

output="image"

PYTHONPATH=$PYTHONPATH:~/Project
export PYTHONPATH

python parsing.py -output $output -sampling 0 -ratio 0.6
echo
echo
python encoding.py -input $output -output image_vector -w2v 0 -ver 1
echo
