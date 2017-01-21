#!/bin/sh

rm -rf temp/
mkdir temp/
rm -rf *~
python experiment.py > result/experiment.txt


rm -rf temp/
