#/bin/bash
#cat *.txt > mergd

id=0
epoch=0
hidden=0

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1


if [[ $# -le 0 ]]; then
	echo "  Usage : ./run.sh --ratio 0.7 [raw_corpus]"
	echo "  Please must input raw_corpus file name !!"
	echo "  -c : set Top x% of abbreviation words  (default is 0.5)"
	exit 1
fi


echo
echo $id + "_h_" + $hidden + "_ep_" + $epoch
echo
python training.py -id $id -epoch $epoch -hidden $hidden > result/reverse_h_00_ep_20000_w2v


#echo
#echo "h_00_ep_w2v"
#echo
#python training.py -w2v 1 -epoch 20000 -hidden 0 > result/h_00_ep_20000_w2v

#echo
#echo "h_01_ep_w2v"
#echo
#python training.py -w2v 1 -epoch 10000 -hidden 1 > result/h_01_ep_10000_w2v

#echo
#echo "h_02_ep_w2v"
#echo
#python training.py -w2v 1 -epoch 10000 -hidden 2 > result/h_02_ep_10000_w2v

