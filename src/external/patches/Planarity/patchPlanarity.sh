#!/bin/sh

# This script loads planarity 

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
PLANARITY_FILENAME=planarity.zip
PLANARITY_URL=http://jgaa.info/accepted/2004/BoyerMyrvold2004.8.3/

PLANARITY_SOURCE_FOLDER=../../planarity.src-patched/

# check if destination folder already exists
if [ -e "$PLANARITY_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download PLANARITY
echo "Getting $PLANARITY_FILENAME from $PLANARITY_URL ..."
if [ -e "$ZIP_FOLDER$PLANARITY_FILENAME" ]
then
    echo "$PLANARITY_FILENAME already exists, skipping download."
else
    wget -q $PLANARITY_URL$PLANARITY_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$PLANARITY_FILENAME" ]
then :
else
    echo "Couldn't download $PLANARITY_FILENAME. Check if $PLANARITY_URL$PLANARITY_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $PLANARITY_FILENAME"
mkdir $PLANARITY_SOURCE_FOLDER
unzip -juqn  $ZIP_FOLDER$PLANARITY_FILENAME -d $PLANARITY_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $PLANARITY_FILENAME."
    exit 1
fi
