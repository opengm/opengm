#!/bin/sh

# This script loads QPBO-LIB v 1.3 from http://pub.ist.ac.at/~vnk/software.html and applies a patch to make QPBO-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
QPBO_FILENAME=QPBO-v1.3.src.tar.gz
QPBO_URL=http://pub.ist.ac.at/~vnk/software/
QPBO_SOURCE_FOLDER=../../QPBO-v1.3.src-patched/
QPBO_PATCH_NAME=QPBO-v1.3.patch

# check if destination folder already exists
if [ -e "$QPBO_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
else
    mkdir $QPBO_SOURCE_FOLDER
fi

# download QPBO
echo "Getting $QPBO_FILENAME from $QPBO_URL ..."
if [ -e "$ZIP_FOLDER$QPBO_FILENAME" ]
then
    echo "$QPBO_FILENAME already exists, skipping download."
else
    wget -q $QPBO_URL$QPBO_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$QPBO_FILENAME" ]
then :
else
    echo "Couldn't download $QPBO_FILENAME. Check if $QPBO_URL$QPBO_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $QPBO_FILENAME"
tar xfz $ZIP_FOLDER$QPBO_FILENAME -C $QPBO_SOURCE_FOLDER --strip-components 1
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $QPBO_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $QPBO_SOURCE_FOLDER -p1 < $PATCH_FOLDER$QPBO_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi