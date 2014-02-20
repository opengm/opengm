#!/bin/sh

# This script loads IBFS-LIB  from http://www.cs.tau.ac.il/~sagihed/ibfs/ and applies a patch to make IBFS-LIB workable with openGM.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
#IBFS_FILENAME=ibfs.zip
#IBFS_URL=http://www.cs.tau.ac.il/~sagihed/ibfs/
IBFS_FILENAME=ibfs.tar
IBFS_URL=http://www.cs.tau.ac.il/~sagihed/ibfs/download/

IBFS_SOURCE_FOLDER=../../ibfs.src-patched/
IBFS_PATCH_NAME=ibfs.patch

# check if destination folder already exists
if [ -e "$IBFS_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download IBFS
echo "Getting $IBFS_FILENAME from $IBFS_URL ..."
if [ -e "$ZIP_FOLDER$IBFS_FILENAME" ]
then
    echo "$IBFS_FILENAME already exists, skipping download."
else
    wget -q $IBFS_URL$IBFS_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$IBFS_FILENAME" ]
then :
else
    echo "Couldn't download $IBFS_FILENAME. Check if $IBFS_URL$IBFS_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $IBFS_FILENAME"
mkdir $IBFS_SOURCE_FOLDER
tar xvf  $ZIP_FOLDER$IBFS_FILENAME -C $IBFS_SOURCE_FOLDER
#unzip -juqn $ZIP_FOLDER$IBFS_FILENAME -d $IBFS_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $IBFS_FILENAME."
    exit 1
fi

## run patch
#echo "Patching files..."
#patch -s -d $IBFS_SOURCE_FOLDER -p1 < $PATCH_FOLDER$IBFS_PATCH_NAME -N -r -
#if [ "$?" = "0" ]
#then 
#    echo "Patching files done"
#else
#    echo "Couldn't run patch"
#    exit 1
#fi