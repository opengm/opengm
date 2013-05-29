#!/bin/sh

# This script loads FastPD from http://www.csd.uoc.gr/~komod/FastPD/ and applies a patch to make FastPD-LIB workable with openGM.
# Important: Registration is required to get the FastPD source. 
# Thus the user needs to provide a valid download link for this script to work properly
# See README.txt for details.

USER_PROVIDED_URL=$1
ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
FastPD_FILENAME=FastPD_DemoVersion.zip
FastPD_URL=http://www.csd.uoc.gr/~komod/FastPD/
FastPD_SOURCE_FOLDER=../../FastPD.src-patched/
FastPD_PATCH_NAME=FastPD.patch

# check if destination folder already exists
if [ -e "$FastPD_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download FastPD
echo "Getting $FastPD_FILENAME from $FastPD_URL ..."
if [ -e "$ZIP_FOLDER$FastPD_FILENAME" ]
then
    echo "$FastPD_FILENAME already exists, skipping download."
else
    if [ "$USER_PROVIDED_URL" = "" ]
    then
        echo "Registration is required to get the FastPD source."
        echo "Thus the user needs to provide a valid download link via CMake (Set FASTPD_URL accordingly)"
        echo "Skipping FastPD patch"
        exit 0
    else
        wget -q $USER_PROVIDED_URL -O $FastPD_FILENAME
        mv $FastPD_FILENAME $ZIP_FOLDER
    fi
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$FastPD_FILENAME" ]
then :
else
    echo "Couldn't download $FastPD_FILENAME. Check if $USER_PROVIDED_URL is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $FastPD_FILENAME"
unzip -juqn $ZIP_FOLDER$FastPD_FILENAME -d $FastPD_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $FastPD_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $FastPD_SOURCE_FOLDER -p1 < $PATCH_FOLDER$FastPD_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi
