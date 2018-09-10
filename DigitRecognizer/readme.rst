#!/bin/bash

####If the kaggle api is available, you can run:
kaggle competitions download -c digit-recognizer

###If the kaggle api is not available, go to https://www.kaggle.com/c/digit-recognizer/data
###After downloading the data, save it in the "data" directory:
mkdir -p ./data
mv test.csv ./data/
mv train.csv
rm sample_submission.csv

###this readme can be run: "$ bash ./readme.rst"
