#!/bin/sh
folder=$1
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

rename() {
	local cnt=0
	local tmp="tmp"

	for dir in $cmd_dir
	do
		cd $folder/$dir

		for file in $(ls)
		do
			cnt=$(($cnt+1))
			mv $file $tmp

			if [ $cnt -ge 100 ]; then
			    mv $tmp $cnt.jpg
            elif [ $cnt -ge 10 ]; then
			    mv $tmp 0$cnt.jpg
            else
			    mv $tmp 00$cnt.jpg
            fi

		done
	
		cd ../../..
		cnt=0
	done
}


count
rename
echo "Finish!"
count
