#!/bin/bash
echo "*************************************************************************"
echo "              Testing Lexicons in Supervised method                    "
echo "*************************************************************************"
echo "==================== Log Linear Classification =========================="
echo "*************************************************************************"
echo "                              WEKA Lexicons                              "
echo "------------------------------ ED tweets --------------------------------"
python liblinear.py ../data/libdata/semeval/libformat/train.expand.ed ../data/libdata/semeval/libformat/dev.expand.ed ../data/libdata/semeval/libformat/test.expand.ed ../data/libdata/semeval/predict/expand.ed ../data/libdata/semeval/result/expand.ed > ./model/temp
cat ../data/libdata/semeval/result/expand.ed
echo "------------------------------ STS tweets --------------------------------"
python liblinear.py ../data/libdata/semeval/libformat/train.expand.sts ../data/libdata/semeval/libformat/dev.expand.sts ../data/libdata/semeval/libformat/test.expand.sts ../data/libdata/semeval/predict/expand.sts ../data/libdata/semeval/result/expand.sts > ./model/temp
cat ../data/libdata/semeval/result/expand.sts
echo "#########################################################################"
echo "                              HIT Lexicons                              "
python liblinear.py ../data/libdata/semeval/libformat/train.sspe ../data/libdata/semeval/libformat/dev.sspe ../data/libdata/semeval/libformat/test.sspe ../data/libdata/semeval/predict/sspe ../data/libdata/semeval/result/sspe > ./model/temp
cat ../data/libdata/semeval/result/sspe
echo "#########################################################################"
echo "                              NRC Lexicons                               "
echo "--------------------- Hashtag tweets (unigram) --------------------------"
python liblinear.py ../data/libdata/semeval/libformat/train.so.ht ../data/libdata/semeval/libformat/dev.so.ht ../data/libdata/semeval/libformat/test.so.ht ../data/libdata/semeval/predict/so.ht ../data/libdata/semeval/result/so.ht > ./model/temp
cat ../data/libdata/semeval/result/so.ht
echo "--------------------- Emoticon tweets (unigram) --------------------------"
python liblinear.py ../data/libdata/semeval/libformat/train.so.emo ../data/libdata/semeval/libformat/dev.so.emo ../data/libdata/semeval/libformat/test.so.emo ../data/libdata/semeval/predict/so.emo ../data/libdata/semeval/result/so.emo > ./model/temp
cat ../data/libdata/semeval/result/so.emo
echo "#########################################################################"
echo "                              NN Lexicons                               "
python liblinear.py ../data/libdata/semeval/libformat/train.sowe ../data/libdata/semeval/libformat/dev.sowe ../data/libdata/semeval/libformat/test.sowe ../data/libdata/semeval/predict/sowe ../data/libdata/semeval/result/sowe > ./model/temp
cat ../data/libdata/semeval/result/sowe
echo "*************************************************************************"
echo "                                 END                                     "
echo "*************************************************************************"