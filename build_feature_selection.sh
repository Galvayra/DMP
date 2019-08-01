#/bin/bash

input=""
output=""
vector="modeling/vectors/"
fs_name=""
ntree=""
show=0
softmax=0

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1

echo
echo "################# vectorization of feature selection #################"
echo
python encoding.py -input "$input" -output "$output" -ver 2 -w2v 0 -softmax "$softmax"

echo
echo "################# get importance of features #################"
echo

if [ "$fs_name" == "" ]; then
    python extract_feature.py -vector "modeling/vectors/""$output" -output "$output" -ntree "$ntree" -show "$show"
else
    python extract_feature.py -vector "modeling/vectors/""$output" -output "$fs_name" -ntree "$ntree" -show "$show"
fi
