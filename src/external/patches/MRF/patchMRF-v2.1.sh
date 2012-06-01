#!/bin/sh

# This script loads MRF-LIB v 2.1 from http://vision.middlebury.edu/MRF/code/ and applies a patch to make MRF-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
MRFLIB_FILENAME=MRF2.1.zip
MRFLIB_URL=http://vision.middlebury.edu/MRF/code/
MRFLIB_SOURCE_FOLDER=../../MRF-v2.1.src-patched/
MRFLIB_PATCH_NAME=MRF2.1.patch
MRFLIB_REPLACE_FILE=typeTruncatedQuadratic2D.h
MRFLIB_REPLACE_FILE_BACKUP=typeTruncatedQuadratic2D_old.h

TRWS_FILENAME=TRW_S-v1.3.zip
TRWS_SUBFOLDER=TRW_S-v1.3/
TRWS_URL=http://ftp.research.microsoft.com/downloads/dad6c31e-2c04-471f-b724-ded18bf70fe3/

# check if destination folder already exists
if [ -e "$MRFLIB_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download MRF-LIB
echo "Getting $MRFLIB_FILENAME from $MRFLIB_URL ..."
if [ -e "$ZIP_FOLDER$MRFLIB_FILENAME" ]
then
    echo "$MRFLIB_FILENAME already exists, skipping download."
else
    wget -q $MRFLIB_URL$MRFLIB_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$MRFLIB_FILENAME" ]
then :
else
    echo "Couldn't download $MRFLIB_FILENAME. Check if $MRFLIB_URL$MRFLIB_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $MRFLIB_FILENAME"
unzip -juqn $ZIP_FOLDER$MRFLIB_FILENAME -d $MRFLIB_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $MRFLIB_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $MRFLIB_SOURCE_FOLDER -p1 < $PATCH_FOLDER$MRFLIB_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi


# replace typeTruncatedQuadratic2D.h
echo "Replacing file $MRFLIB_REPLACE_FILE to enable support for TRW-S with truncated L2 terms."
mv $MRFLIB_SOURCE_FOLDER$MRFLIB_REPLACE_FILE $MRFLIB_SOURCE_FOLDER$MRFLIB_REPLACE_FILE_BACKUP
if [ "$?" = "0" ]
then :
else
    echo "Couldn't rename file $MRFLIB_REPLACE_FILE."
    exit 1
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
unzip -juqn $ZIP_FOLDER$TRWS_FILENAME $TRWS_SUBFOLDER$MRFLIB_REPLACE_FILE -d $MRFLIB_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $MRFLIB_REPLACE_FILE from file $TRWS_FILENAME."
    exit 1
fi



