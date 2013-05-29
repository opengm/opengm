#!/bin/sh

# This script loads grante v 1.0 from http://www.nowozin.net/sebastian/grante/ and applies a patch to make grante workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
GRANTE_FILENAME=grante-1.0.zip
GRANTE_URL=http://ftp.research.microsoft.com/downloads/b51cf9cc-9297-49d4-af3c-5314b99e4240/
GRANTE_SOURCE_FOLDER=../../grante-v1.0.src-patched/
GRANTE_PATCH_NAME=grante-1.0.patch

# check if destination folder already exists
if [ -e "$GRANTE_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download grante
echo "Getting $GRANTE_FILENAME from $GRANTE_URL ..."
if [ -e "$ZIP_FOLDER$GRANTE_FILENAME" ]
then
    echo "$GRANTE_FILENAME already exists, skipping download."
else
    wget -q $GRANTE_URL$GRANTE_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$GRANTE_FILENAME" ]
then :
else
    echo "Couldn't download $GRANTE_FILENAME. Check if $GRANTE_URL$GRANTE_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $GRANTE_FILENAME"
unzip -juqn $ZIP_FOLDER$GRANTE_FILENAME -d $GRANTE_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $GRANTE_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $GRANTE_SOURCE_FOLDER -p1 < $PATCH_FOLDER$GRANTE_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi
