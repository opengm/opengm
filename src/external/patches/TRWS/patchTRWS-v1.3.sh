#!/bin/sh

# This script loads TRW-S-LIB v 1.3 from http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/ and applies a patch to make TRW-S-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
TRWS_FILENAME=TRW_S-v1.3.zip
TRWS_URL=http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/
TRWS_SOURCE_FOLDER=../../TRWS-v1.3.src-patched/
TRWS_PATCH_NAME=TRWS-v1.3.patch

# check if destination folder already exists
if [ -e "$TRWS_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download TRWS
echo "Getting $TRWS_FILENAME from $TRWS_URL ..."
if [ -e "$ZIP_FOLDER$TRWS_FILENAME" ]
then
    echo "$TRWS_FILENAME already exists, skipping download."
else
    wget -q $TRWS_URL$TRWS_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$TRWS_FILENAME" ]
then :
else
    echo "Couldn't download $TRWS_FILENAME. Check if $TRWS_URL$TRWS_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $TRWS_FILENAME"
unzip -juqn $ZIP_FOLDER$TRWS_FILENAME -d $TRWS_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $TRWS_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $TRWS_SOURCE_FOLDER -p1 < $PATCH_FOLDER$TRWS_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi