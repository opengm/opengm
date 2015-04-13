#!/bin/sh

# This script loads blossom5 

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
BLOSSOM5_FILENAME=blossom5-v2.05.src.tar.gz
BLOSSOM5_URL=http://pub.ist.ac.at/~vnk/software/

BLOSSOM5_SOURCE_FOLDER=../../blossom5.src-patched/

# check if destination folder already exists
if [ -e "$BLOSSOM5_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download BLOSSOM5
echo "Getting $BLOSSOM5_FILENAME from $BLOSSOM5_URL ..."
if [ -e "$ZIP_FOLDER$BLOSSOM5_FILENAME" ]
then
    echo "$BLOSSOM5_FILENAME already exists, skipping download."
else
    wget -q $BLOSSOM5_URL$BLOSSOM5_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$BLOSSOM5_FILENAME" ]
then :
else
    echo "Couldn't download $BLOSSOM5_FILENAME. Check if $BLOSSOM5_URL$BLOSSOM5_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $BLOSSOM5_FILENAME"
mkdir $BLOSSOM5_SOURCE_FOLDER
tar -xvf  $ZIP_FOLDER$BLOSSOM5_FILENAME -C $BLOSSOM5_SOURCE_FOLDER 
mv $BLOSSOM5_SOURCE_FOLDER/blossom5-v2.05.src/*  $BLOSSOM5_SOURCE_FOLDER
rmdir $BLOSSOM5_SOURCE_FOLDER/blossom5-v2.05.src
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $BLOSSOM5_FILENAME."
    exit 1
fi
