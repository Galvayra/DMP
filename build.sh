#/bin/bash

parsing=0
except=0
all=0

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
	if [ $except -eq 0 ]; then
		python encoding.py -fold 1 -output all
	else
		python encoding.py -fold 1 -output all_+$except
	fi
fi

echo
echo "---------------------------------"
echo

if [ $except -eq 0 ]; then
	python encoding.py -target s -fold 1 -input sepsis.csv -output sepsis
else
	python encoding.py -target s -fold 1 -input sepsis.csv -output sepsis_+$except
fi
echo
echo "---------------------------------"
echo

if [ $except -eq 0 ]; then
	python encoding.py -target b -fold 1 -input bacteremia.csv -output bacteremia
else
	python encoding.py -target b -fold 1 -input bacteremia.csv -output bacteremia_+$except
fi
echo
echo "---------------------------------"
echo

if [ $except -eq 0 ]; then
	python encoding.py -target p -fold 1 -input pneumonia.csv -output pneumonia
else
	python encoding.py -target p -fold 1 -input pneumonia.csv -output pneumonia_+$except
fi


