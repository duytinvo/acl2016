#!/bin/bash
echo "*************************************************************************"
echo "              Testing Lexicons in Supervised method                    "
echo "*************************************************************************"
echo "==================== Feature Extraction ================================="
echo "*************************************************************************"
echo "                              WEKA Lexicons                              "
echo "------------------------------ ED tweets --------------------------------"
python features.py ../data/semeval13/train ../data/semeval13/dev ../data/semeval13/test ../data/lexicons/expand_felipe/EDLex.csv .expand.ed
echo "------------------------------ STS tweets --------------------------------"
python features.py ../data/semeval13/train ../data/semeval13/dev ../data/semeval13/test ../data/lexicons/expand_felipe/STSLex.csv .expand.sts
echo "#########################################################################"
echo "                              HIT Lexicons                              "
python features.py ../data/semeval13/train ../data/semeval13/dev ../data/semeval13/test ../data/lexicons/sspe_tang/ .sspe
echo "#########################################################################"
echo "                              NRC Lexicons                               "
echo "--------------------- Hashtag tweets (unigram) --------------------------"
python features.py ../data/semeval13/train ../data/semeval13/dev ../data/semeval13/test ../data/lexicons/so_mohammad/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt .so.ht
echo "--------------------- Emoticon tweets (unigram) --------------------------"
python features.py ../data/semeval13/train ../data/semeval13/dev ../data/semeval13/test ../data/lexicons/so_mohammad/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt .so.emo
echo "#########################################################################"
echo "                              NN Lexicons                               "
python features.py ../data/semeval13/train ../data/semeval13/dev ../data/semeval13/test ../data/lexicons/mylexicon/epoch_5Words_current.npy .sowe
echo "*************************************************************************"
echo "                                 END                                     "
echo "*************************************************************************"









