#/bin/bash

parsing=0
id=""
all=0
fold=5

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1


if [ $parsing -eq 1 ]; then
	echo
	echo "============ Parsing ============"
	echo

	if [ $all -eq 1 ]; then
		python parsing.py
	fi
	echo
	echo "---------------------------------"
	echo

	python parsing.py -target s -output sepsis.csv
	echo
	echo "---------------------------------"
	echo

	python parsing.py -target b -output bacteremia.csv
	echo
	echo "---------------------------------"
	echo

	python parsing.py -target p -output pneumonia.csv	
fi

echo
echo
echo "============ Encoding ============"
echo

if [ $all -eq 1 ]; then
	if [ -n $id ]; then
		python encoding.py -fold $fold -output all
	else
		python encoding.py -fold $fold -output all_except
	fi
fi

echo
echo "---------------------------------"
echo

if [ -n $id ]; then
	python encoding.py -target s -fold $fold -input sepsis.csv -output sepsis
else
	python encoding.py -target s -fold $fold -input sepsis.csv -output sepsis_except
fi
echo
echo "---------------------------------"
echo

if [ -n $id ]; then
	python encoding.py -target b -fold $fold -input bacteremia.csv -output bacteremia
else
	python encoding.py -target b -fold $fold -input bacteremia.csv -output bacteremia_except
fi
echo
echo "---------------------------------"
echo

if [ -n $id ]; then
	python encoding.py -target p -fold $fold -input pneumonia.csv -output pneumonia
else
	python encoding.py -target p -fold $fold -input pneumonia.csv -output pneumonia_except
fi

