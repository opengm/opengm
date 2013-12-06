#!/bin/sh

# This script loads DAOOPT from https://github.com/joergkappes/daoopt/archive/master.zip 

ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
DAOOPT_FILENAME=master.zip
DAOOPT_URL=https://github.com/joergkappes/daoopt/archive/
DAOOPT_SOURCE_FOLDER=../../DAOOPT.src/

# check if destination folder already exists
if [ -e "$DAOOPT_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
fi

# download DAOOPT
echo "Getting $DAOOPT_FILENAME from $DAOOPT_URL ..."
if [ -e "$ZIP_FOLDER$DAOOPT_FILENAME" ]
then
    echo "$DAOOPT_FILENAME already exists, skipping download."
else
    wget --no-check-certificate -q $DAOOPT_URL$DAOOPT_FILENAME -P $ZIP_FOLDER
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$DAOOPT_FILENAME" ]
then :
else
    echo "Couldn't download $DAOOPT_FILENAME. Check if $DAOOPT_URL$DAOOPT_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $DAOOPT_FILENAME"
unzip -uqn $ZIP_FOLDER$DAOOPT_FILENAME -d $DAOOPT_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $DAOOPT_FILENAME."
    exit 1
fi

# remove container folder
echo "Moving files"
mv $DAOOPT_SOURCE_FOLDER/daoopt-master/* $DAOOPT_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't move files."
    exit 1
fi
rm -r $DAOOPT_SOURCE_FOLDER/daoopt-master
if [ "$?" = "0" ]
then :
else
    echo "Couldn't remove folder."
    exit 1
fi
