# acl2016
This code is used for the paper "Don't Count, Predict! An Automatic Approach to Learning Sentiment Lexicons for Short Text" (ACL2016-short paper)

This repository consists of three folders:
  - learn_lexicon: scripts to learn lexicons
  - test_lexicon: scripts to test ours learned lexicons
  - lexicons: sentiment lexicons in English and Arabic, which are learned by our models.
  
To use our lexicons:
  - English and Arabic lexicons in "lexicons" folder.
  - Data format: word + '\t' + sentiment score + '\n'
  
To compare our lexicons with available lexicons:
  - Change current directory to folder "acl2016/test_lexicon/scripts"
  - Change all *.sh file to 755 (e.g. chmod 755 *.sh)
  - Change all files in folder liblinear to 755 (e.g. chmod 755 ./liblinear/*)
  - For testing English lexicons:
      + Extract features: ./run_features.sh
      + Classification: ./run_liblinear.sh
  - For testing Arabic lexicons:
      + Extract features: ./run_features_ar.sh
      + Classification: ./run_liblinear_ar.sh
      
To run the our models to learn lexicons:
  - Download the data from "https://drive.google.com/folderview?id=0Bys5jWIGhUopakxaNjlPWG10RDA&usp=sharing" 
  - Replace to data folder
  - Change current directory to folder "acl2016/learn_lexicon/script"
  - Run:
      + English model: python lexicon-english.py
      + Arabic model: python lexicon-arabic.py
	  
TO LEARN MODEL BY YOUR OWN DATA:
  - Process your data:
    + Example: python process.py ../data/alexgo/raw/metatweets ../data/alexgo/processed/info.tw ../data/alexgo/processed/process.tw
    + Raw data format: labels (1 or 0) + ' ' + tweet +'\n'
  - Modify the script by changing the inputs to ../data/alexgo/processed/info.tw and  ../data/alexgo/processed/process.tw
  - Run model
