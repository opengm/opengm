#!/bin/sh

# This script loads GCO-LIB v 3.0 from http://vision.csd.uwo.ca/code/ and applies a patch to make GCO-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
GCO_FILENAME=gco-v3.0.zip
GCO_URL=http://vision.csd.uwo.ca/code/
GCO_SOURCE_FOLDER=../../GCO-v3.0.src-patched/
#GCO_PATCH_NAME=GCO-v3.0.patch
GCO_PATCH_NAME=gco-3.0.1.dif

# check if destination folder already exists
if [ -e "$GCO_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download GCO
echo "Getting $GCO_FILENAME from $GCO_URL ..."
if [ -e "$ZIP_FOLDER$GCO_FILENAME" ]
then
    echo "$GCO_FILENAME already exists, skipping download."
else
    wget -q $GCO_URL$GCO_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$GCO_FILENAME" ]
then :
else
    echo "Couldn't download $GCO_FILENAME. Check if $GCO_URL$GCO_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $GCO_FILENAME"
unzip -uqn $ZIP_FOLDER$GCO_FILENAME -d $GCO_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $GCO_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch --verbose -s -d $GCO_SOURCE_FOLDER -p1 < $PATCH_FOLDER$GCO_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi
