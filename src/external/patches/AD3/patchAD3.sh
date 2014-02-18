#!/bin/sh


ZIP_FOLDER=../zip_files/
PATCH_FOLDER=./

AD3_FILENAME=master.zip
AD3_FILENAME2=masterAd3.zip
AD3_URL=http://github.com/andre-martins/AD3/archive/

AD3_MASTER=AD3-master



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
echo "Getting $AD3_FILENAME2 from $AD3_URL ..."
if [ -e "$ZIP_FOLDER$AD3_FILENAME2" ]
then
    echo "$AD3_FILENAME already exists, skipping download."
else
	wget --no-check-certificate -q $AD3_URL$AD3_FILENAME  -O $ZIP_FOLDER$AD3_FILENAME2
    #wget --output-document=masterAd3.zip -q $AD3_URL$AD3_FILENAME -P $ZIP_FOLDER 
fi

# check if download was successful
if [ -e "$ZIP_FOLDER$AD3_FILENAME2" ]
then :
else
    echo "Couldn't download $AD3_FILENAME. Check if $AD3_URL$AD3_FILENAME is reachable!"
    rmdir $AD3_SOURCE_FOLDER
    exit 1
fi

# extract files
echo "Extracting files from $AD3_FILENAME2"
#tar xfz $ZIP_FOLDER$AD3_FILENAME -C $AD3_SOURCE_FOLDER --strip-components 1
unzip  $ZIP_FOLDER$AD3_FILENAME2 -d $AD3_SOURCE_FOLDER
mv $AD3_SOURCE_FOLDER$AD3_MASTER/* $AD3_SOURCE_FOLDER
if [ "$?" = "0" ]
then :
else
    echo "Couldn't extract $AD3_FILENAME2." 
    rmdir $AD3_SOURCE_FOLDER
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
    rmdir $AD3_SOURCE_FOLDER
    exit 1
fi