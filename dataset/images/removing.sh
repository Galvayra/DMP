#!/bin/sh
folder=$1
cut_len=$2
cmd_dir=$(ls $folder | grep -v '[^0-9_]')

count() {
	local sum=0
	local cnt=0
	
	for dir in $cmd_dir
	do
		cd $folder/$dir
		cnt=$(ls | wc -l)
	
		sum=$(($sum + $cnt))	
		cd ../../..
	done
	
	echo "total  count = ${sum}"
	echo
}

remove() {
	local cnt=0
	local tmp="tmp"
	for dir in $cmd_dir
	do
		cd $folder/$dir

		for file in $(ls)
		do
		    local ff=`echo $file | cut -d '.' -f1`

			if [ $ff -gt $cut_len ]; then
			    rm $file
			fi
		done
	
		cd ../../..
		cnt=0
	done
}


count
remove
echo "Finish!"
count
