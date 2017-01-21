#Introduction

This is the source code for predicting the amount of CO2 emission from a power plant in a research project that is sponsored by Nigerian Liquefied Natural Gas Treatment plant (Train 4). All codes are in Python programming language and the result of the research will be published in a upcoming journal article which will be co-authored by me. The models consist of MARS (Multivariate adaptive regression splines), Extra Random Trees, Neural Network, and ANFIS (Adaptive neuro fuzzy inference system).

All codes are in Python programming

The required libraries are installed

Install the scikit-fuzzy package by following the instructions here: https://github.com/scikit-fuzzy/scikit-fuzzy

This is accompanied with install the ANFIS package

pip install git+https://github.com/scikit-fuzzy/scikit-fuzzy

pip install git+https://github.com/timesofbadri/anfis

pip install -U scikit-learn

pip install git+https://github.com/scikit-learn-contrib/py-earth


Model comparison was done using Regression Error Characteristics Curves.

The file structure is 

    /src

        /data  

        /pictures

        /result

            experiment.txt

            output.txt

        process.py  

        runexp.sh

        experiment.py  

        predict.py 

        run.sh

    README.md

The data set is stored in the data folder.

To perform all the required hyperparameter search for the best parameters of the models in use.

$ chmod u+x runexp.sh

$ ./runexp.sh

The output is available in result folder in the file named experiment.txt.

To run the actual code

$ chmod u+x run.sh

$ ./run.sh


5-fold cross validation was used to prevent overfitting, thereby making our experiment very realistic.

#Licence
All rights reserved as no part of this code can be used in any form (research or academic) until the paper has been published.

#References

Friedman, J. (1991). Multivariate adaptive regression splines. The annals of statistics, 19(1), 1–67. http://www.jstor.org/stable/10.2307/2241837

Chakrapani, K. S. (2010). Adaptive Neuro-Fuzzy Inference System based
actal Image Compression. International Journal on Signal & Image
 Processing , 1, 6. 

Bi, J and Bennett, K (2003). Regression Error Characteristic Curves


