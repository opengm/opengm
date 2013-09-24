#!/bin/sh


ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./
AD3_FILENAME=AD3-2.0.2.tar.gz
AD3_URL=http://www.cs.cmu.edu/~afm/AD3/
AD3_SOURCE_FOLDER=../../AD3-patched/
AD3_PATCH_NAME=AD3.patch

# check if destination folder already exists
if [ -e "$AD3_SOURCE_FOLDER" ]
then 
	echo "Source folder already exists, skipping patch."
	exit 0
else
	mkdir $AD3_SOURCE_FOLDER
fi

# download AD3
echo "Getting $AD3_FILENAME from $AD3_URL ..."
if [ -e "$ZIP_FOLDER$AD3_FILENAME" ]
then
    echo "$AD3_FILENAME already exists, skipping download."
else
	wget -q $AD3_URL$AD3_FILENAME -P $ZIP_FOLDER
    #wget --output-document=masterAd3.zip -q $AD3_URL$AD3_FILENAME -P $ZIP_FOLDER 
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$AD3_FILENAME" ]
then :
else
    echo "Couldn't download $AD3_FILENAME. Check if $AD3_URL$AD3_FILENAME is reachable!"
    exit 1
fi

# extract files
echo "Extracting files from $AD3_FILENAME"
tar xfz $ZIP_FOLDER$AD3_FILENAME -C $AD3_SOURCE_FOLDER --strip-components 1
#unzip -juqn $ZIP_FOLDER$AD3_FILENAME -d $AD3_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $AD3_FILENAME."
    exit 1
fi

# run patch
echo "Patching files..."
patch -s -d $AD3_SOURCE_FOLDER -p1 < $PATCH_FOLDER$AD3_PATCH_NAME -N -r -
if [ "$?" = "0" ]
then 
    echo "Patching files done"
else
    echo "Couldn't run patch"
    exit 1
fi