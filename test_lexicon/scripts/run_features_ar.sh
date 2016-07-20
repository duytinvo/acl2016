#!/bin/bash
echo "*************************************************************************"
echo "              Testing Lexicons in Supervised method                    "
echo "*************************************************************************"
echo "==================== Feature Extraction ================================="
echo "#########################################################################"
echo "                              NRC Lexicons                               "
echo "--------------------- Hashtag tweets (unigram) --------------------------"
echo "----------------------------  Balanced dataset --------------------------"
python features_ar.py  balanced ../data/lexicons/so-lexicon-ar/Arabic_Hashtag_Lexicon.txt .so.ht
echo "--------------------------- Unbalanced dataset --------------------------"
python features_ar.py  unbalanced ../data/lexicons/so-lexicon-ar/Arabic_Hashtag_Lexicon.txt .so.ht
echo "--------------------- Emoticon tweets (unigram) --------------------------"
echo "----------------------------  Balanced dataset --------------------------"
python features_ar.py balanced ../data/lexicons/so-lexicon-ar/Arabic_Emoticon_Lexicon.txt .so.Emo
echo "--------------------------- Unbalanced dataset --------------------------"
python features_ar.py  unbalanced ../data/lexicons/so-lexicon-ar/Arabic_Emoticon_Lexicon.txt .so.Emo
echo "                              NN Lexicons                               "
echo "----------------------------  Balanced dataset --------------------------"
python features_ar.py  balanced ../data/lexicons/mylexicon-ar/epoch_5Words_current.npy .sowe
echo "--------------------------- Unbalanced dataset --------------------------"
python features_ar.py  unbalanced ../data/lexicons/mylexicon-ar/epoch_5Words_current.npy .sowe
echo "*************************************************************************"
echo "                                 END                                     "
echo "*************************************************************************"