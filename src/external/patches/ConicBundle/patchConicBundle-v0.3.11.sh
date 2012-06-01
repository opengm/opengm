#!/bin/sh

# This script loads ConicBundle-LIB v 0.3.11 from http://www-user.tu-chemnitz.de/~helmberg/ConicBundle/ applies a patch to make ConicBundle-LIB workable with openGM and builds it.
# See README.txt for details.

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
ConicBundle_FILENAME=CB_v0.3.11.tgz
ConicBundle_URL=http://www-user.tu-chemnitz.de/~helmberg/ConicBundle/
ConicBundle_SOURCE_FOLDER=../../ConicBundle-v0.3.11.src-patched/
ConicBundle_PATCH_NAME=ConicBundle-v0.3.11.patch

# check if destination folder already exists
if [ -e "$ConicBundle_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
else
    mkdir $ConicBundle_SOURCE_FOLDER
fi

# download ConicBundle
echo "Getting $ConicBundle_FILENAME from $ConicBundle_URL ..."
if [ -e "$ZIP_FOLDER$ConicBundle_FILENAME" ]
then
    echo "$ConicBundle_FILENAME already exists, skipping download."
else
    wget -q $ConicBundle_URL$ConicBundle_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$ConicBundle_FILENAME" ]
then :
else
    echo "Couldn't download $ConicBundle_FILENAME. Check if $ConicBundle_URL$ConicBundle_FILENAME is reachable!"
    exit 1
fi

# extract files

echo "Extracting files from $ConicBundle_FILENAME"
tar xfz $ZIP_FOLDER$ConicBundle_FILENAME -C $ConicBundle_SOURCE_FOLDER --strip-components 1
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $ConicBundle_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $ConicBundle_SOURCE_FOLDER -p1 < $PATCH_FOLDER$ConicBundle_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi

# run make
echo "Running make..."
make -s -C $ConicBundle_SOURCE_FOLDER
if [ "$?" = "0" ]
then 
    echo "Building binaries done"
else
    echo "Couldn't build binaries"
    exit 1
fi