#!/bin/sh

# This script loads ISINF-LIB   

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
ISINF_FILENAME=isinf-1.1.tgz
ISINF_URL=http://nic.schraudolph.org/isinf/
ISINF_SOURCE_FOLDER=../../ISINF-v1.1.src-patched/
ISINF_PATCH_NAME=ISINF.patch

# check if destination folder already exists
if [ -e "$ISINF_SOURCE_FOLDER" ]
then 
    echo "Source folder already exists, skipping patch."
    exit 0
else
    mkdir $ISINF_SOURCE_FOLDER
fi

# download ISINF
echo "Getting $ISINF_FILENAME from $ISINF_URL ..."
if [ -e "$ZIP_FOLDER$ISINF_FILENAME" ]
then
    echo "$ISINF_FILENAME already exists, skipping download."
else
    wget -q $ISINF_URL$ISINF_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$ISINF_FILENAME" ]
then :
else
    echo "Couldn't download $ISINF_FILENAME. Check if $ISINF_URL$ISINF_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $ISINF_FILENAME"
tar -zxvf $ZIP_FOLDER$ISINF_FILENAME -C $ISINF_SOURCE_FOLDER #--strip-components 1
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $ISINF_FILENAME."
    exit 1
fi


echo $ISINF_SOURCE_FOLDER
chmod u+w $ISINF_SOURCE_FOLDER/isinf-1.1/planarity/graph.h
chmod -R u+w $ISINF_SOURCE_FOLDER/


# run patch
echo "Patching files..."
patch -s -d $ISINF_SOURCE_FOLDER -p0 < $PATCH_FOLDER$ISINF_PATCH_NAME -N -r -





if [ "$?" = "0" ]
then 
    echo "Patching files done"
else    
    echo "Couldn't run patch"
    exit 1
fi