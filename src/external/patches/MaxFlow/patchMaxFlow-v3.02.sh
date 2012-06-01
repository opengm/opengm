#!/bin/sh

# This script loads MaxFlow-LIB v 3.02 from http://pub.ist.ac.at/~vnk/software.html and applies a patch to make MaxFlow-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
MaxFlow_FILENAME=maxflow-v3.02.src.tar.gz
MaxFlow_URL=http://pub.ist.ac.at/~vnk/software/
MaxFlow_SOURCE_FOLDER=../../MaxFlow-v3.02.src-patched/
MaxFlow_PATCH_NAME=maxflow-v3.02.patch

# check if destination folder already exists
if [ -e "$MaxFlow_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
else
    mkdir $MaxFlow_SOURCE_FOLDER
fi

# download MaxFlow
echo "Getting $MaxFlow_FILENAME from $MaxFlow_URL ..."
if [ -e "$ZIP_FOLDER$MaxFlow_FILENAME" ]
then
    echo "$MaxFlow_FILENAME already exists, skipping download."
else
    wget -q $MaxFlow_URL$MaxFlow_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$MaxFlow_FILENAME" ]
then :
else
    echo "Couldn't download $MaxFlow_FILENAME. Check if $MaxFlow_URL$MaxFlow_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $MaxFlow_FILENAME"
tar xfz $ZIP_FOLDER$MaxFlow_FILENAME -C $MaxFlow_SOURCE_FOLDER --strip-components 1
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $MaxFlow_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $MaxFlow_SOURCE_FOLDER -p1 < $PATCH_FOLDER$MaxFlow_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi