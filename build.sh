#/bin/bash

parsing=0
id=""
fold=5
sampling=0
target=""
input=dataset
vector=all

. utils/parse_options.sh || echo "Can't find parse_options.sh" | exit 1


if [ "$parsing" -eq 1 ]; then
	echo
	echo "============ Parsing ============"
	echo

    if [ "$target" == "s" ]; then
        python parsing.py -input "$input".csv -output parsing_sepsis.csv -target "$target" -sampling "$sampling"
    elif [ "$target" == "b" ]; then
	    python parsing.py -input "$input".csv -output parsing_bacteremia.csv -target "$target" -sampling "$sampling"
    elif [ "$target" == "p" ]; then
	    python parsing.py -input "$input".csv -output parsing_pneumonia.csv -target "$target" -sampling "$sampling"
	else
        if [ "$sampling" -eq 1 ]; then
	        python parsing.py -input "$input".csv -output parsing_all_sampling.csv -target "$target" -sampling "$sampling"
        else
	        python parsing.py -input "$input".csv -output parsing_all.csv -sampling "$sampling"
        fi
	fi

fi

echo
echo
echo "============ Encoding ============"
echo

if [ "$target" == "s" ]; then
    python encoding.py -input parsing_sepsis.csv -output "$vector" -target "$target"
elif [ "$target" == "b" ]; then
    python encoding.py -input parsing_bacteremia.csv -output "$vector" -target "$target"
elif [ "$target" == "p" ]; then
    python encoding.py -input parsing_pneumonia.csv -output "$vector" -target "$target"
else
    if [ "$sampling" -eq 1 ]; then
        python encoding.py -input parsing_all_sampling.csv -output "$vector"
    else
        python encoding.py -input parsing_all.csv -output "$vector"
    fi
fi