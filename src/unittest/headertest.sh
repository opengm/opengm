#!/bin/sh

for i in `ls *.hxx`
do
  echo "Editing $i ..."
  echo "#include \"$i\" \n int main(){return 0;}" > tmp.cpp
  gcc -o tmp tmp.cpp -Wall
done
