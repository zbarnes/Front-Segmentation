#!/bin/bash
#@author: Zack Barnes
# Bash script to load image files to a 
# txt file named pics to seg

# request name of folder from User
echo -n "Please enter the name the folder containing the time series of images and hit [Enter]: "

# read User input
read FILE


for pics in `ls $FILE`;do

    echo "$pics" >> pics_to_seg.txt

done

# End of script
