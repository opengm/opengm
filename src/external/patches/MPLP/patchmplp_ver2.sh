#!/bin/sh

# This script loads MPLP-LIB from http://cs.nyu.edu/~dsontag/code/ and applies a patch to make MPLP-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
MPLP_FILENAME=mplp_ver2.tgz
MPLP_URL=http://cs.nyu.edu/~dsontag/code/
MPLP_SOURCE_FOLDER=../../mplp_ver2.src-patched/
MPLP_PATCH_NAME=mplp_ver2.patch

# check if destination folder already exists
if [ -e "$MPLP_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
else
    mkdir $MPLP_SOURCE_FOLDER
fi

# download MPLP
echo "Getting $MPLP_FILENAME from $MPLP_URL ..."
if [ -e "$ZIP_FOLDER$MPLP_FILENAME" ]
then
    echo "$MPLP_FILENAME already exists, skipping download."
else
    wget -q $MPLP_URL$MPLP_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$MPLP_FILENAME" ]
then :
else
    echo "Couldn't download $MPLP_FILENAME. Check if $MPLP_URL$MPLP_FILENAME is reachable!"
    exit 1
fi

# extract files

echo "Extracting files from $ConicBundle_FILENAME"
tar xfz $ZIP_FOLDER$MPLP_FILENAME -C $MPLP_SOURCE_FOLDER --strip-components 1
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $MPLP_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $MPLP_SOURCE_FOLDER -p1 < $PATCH_FOLDER$MPLP_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi
