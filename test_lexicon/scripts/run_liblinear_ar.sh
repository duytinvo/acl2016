#!/bin/bash
echo "*************************************************************************"
echo "              Testing Lexicons in Supervised method                    "
echo "*************************************************************************"
echo "==================== Log Linear Classification =========================="
echo "*************************************************************************"
echo "#########################################################################"
echo "                              NRC Lexicons                               "
echo "--------------------- Hashtag tweets (unigram) --------------------------"
echo "----------------------------  Balanced dataset --------------------------"
python liblinear_ar.py ../data/libdata/astd/libformat/train.so.ht.balanced ../data/libdata/astd/libformat/test.so.ht.balanced ../data/libdata/astd/predict/so.ht.balanced ../data/libdata/astd/result/so.ht.balanced > ./model/temp
cat ../data/libdata/astd/result/so.ht.balanced
echo "--------------------------- Unbalanced dataset --------------------------"
python liblinear_ar.py ../data/libdata/astd/libformat/train.so.ht.unbalanced ../data/libdata/astd/libformat/test.so.ht.unbalanced ../data/libdata/astd/predict/so.ht.unbalanced ../data/libdata/astd/result/so.ht.unbalanced > ./model/temp
cat ../data/libdata/astd/result/so.ht.unbalanced
echo "--------------------- Emoticon tweets (unigram) --------------------------"
echo "----------------------------  Balanced dataset --------------------------"
python liblinear_ar.py ../data/libdata/astd/libformat/train.so.Emo.balanced ../data/libdata/astd/libformat/test.so.Emo.balanced ../data/libdata/astd/predict/so.Emo.balanced ../data/libdata/astd/result/so.Emo.balanced > ./model/temp
cat ../data/libdata/astd/result/so.Emo.balanced
echo "--------------------------- Unbalanced dataset --------------------------"
python liblinear_ar.py ../data/libdata/astd/libformat/train.so.Emo.unbalanced ../data/libdata/astd/libformat/test.so.Emo.unbalanced ../data/libdata/astd/predict/so.Emo.unbalanced ../data/libdata/astd/result/so.Emo.unbalanced > ./model/temp
cat ../data/libdata/astd/result/so.Emo.unbalanced
echo "#########################################################################"
echo "                              NN Lexicons                               "
echo "----------------------------  Balanced dataset --------------------------"
python liblinear_ar.py ../data/libdata/astd/libformat/train.sowe.balanced ../data/libdata/astd/libformat/test.sowe.balanced ../data/libdata/astd/predict/sowe.balanced ../data/libdata/astd/result/sowe.balanced > ./model/temp
cat ../data/libdata/astd/result/sowe.balanced
echo "--------------------------- Unbalanced dataset --------------------------"
python liblinear_ar.py ../data/libdata/astd/libformat/train.sowe.unbalanced ../data/libdata/astd/libformat/test.sowe.unbalanced ../data/libdata/astd/predict/sowe.unbalanced ../data/libdata/astd/result/sowe.unbalanced > ./model/temp
cat ../data/libdata/astd/result/sowe.unbalanced
echo "*************************************************************************"
echo "                                 END                                     "
echo "*************************************************************************"
