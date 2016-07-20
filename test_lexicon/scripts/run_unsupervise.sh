#!/bin/bash
echo "*************************************************************************"
echo "              Testing Lexicons in Unsupervised method                    "
echo "*************************************************************************"
echo "                              WEKA Lexicons                              "
echo "------------------------------ ED tweets --------------------------------"
python unsupervise.py ../data/semeval13/semeval.all ../data/lexicons/expand_felipe/EDLex.csv  .expand
echo "------------------------------ STS tweets --------------------------------"
python unsupervise.py ../data/semeval13/semeval.all ../data/lexicons/expand_felipe/STSLex.csv  .expand
echo "#########################################################################"
echo "                              HIT Lexicons                              "
python unsupervise.py ../data/semeval13/semeval.all ../data/lexicons/sspe_tang/  .sspe
echo "#########################################################################"
echo "                              NRC Lexicons                               "
echo "--------------------- Hashtag tweets (unigram) --------------------------"
python unsupervise.py ../data/semeval13/semeval.all ../data/lexicons/so_mohammad/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt  .so
echo "--------------------- Emoticon tweets (unigram) --------------------------"
python unsupervise.py ../data/semeval13/semeval.all ../data/lexicons/so_mohammad/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt  .so
echo "#########################################################################"
echo "                              NN Lexicons                               "
python unsupervise.py ../data/semeval13/semeval.all ../data/lexicons/mylexicon/epoch_5Words_current.npy .sowe
echo "*************************************************************************"
echo "                                 END                                     "
echo "*************************************************************************"