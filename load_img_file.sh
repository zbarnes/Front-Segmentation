#!/bin/bash
#@author: Zack Barnes
# Bash script to load image files to a 
# txt file named pics to seg

# update file path
FILE="/home/usr/Pictures/*"


#prints only the .TIF files
for pics in $FILE.TIF;do

    echo "$pics" >> pics_to_seg.txt

done


