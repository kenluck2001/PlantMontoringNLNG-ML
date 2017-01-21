#!/bin/sh

rm -rf temp/
mkdir temp/
rm -rf *~
python predict.py > result/data.txt
python process.py

rm -rf temp/
