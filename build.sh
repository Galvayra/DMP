#/bin/bash

parsing=0

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1


if [ $parsing -eq 1 ]; then
	echo
	echo "============ Parsing ============"
	echo

	python parsing.py -target s

	echo
	echo "---------------------------------"
	echo

	python parsing.py -target b

	echo
	echo "---------------------------------"
	echo

	python parsing.py -target p	
fi

echo
echo
echo "============ Encoding ============"
echo

python encoding.py -target s -fold 1

echo
echo "---------------------------------"
echo

python encoding.py -target b -fold 1

echo
echo "---------------------------------"
echo

python encoding.py -target p -fold 1
