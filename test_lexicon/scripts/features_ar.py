# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:07:56 2015

@author: duytinvo
"""
import cPickle as pickle
import numpy as np
import codecs as cd
import sys
from AraTweet import AraTweet
         
def load_so(rfile):
    so={}
    with cd.open(rfile,'rb',encoding='utf8') as f:
        for line in f:
            parts=line.strip().split(u'\t')
            if len(parts)==5 and parts[0]!=u'[Arabic Term]':
                so[parts[0]]=float(parts[2])
    so[u'<unk>']=0.
    return so

def readinfo(fname):
    with cd.open(fname, 'rb') as f:
        info=pickle.load(f)
    return info
    
def load_sowe(fname,infofile):
    info=readinfo(infofile)
    word_idx_map=info['vocab']
    wordVectors = np.load(fname)
    embs = {}
    for word in word_idx_map:
        score=wordVectors[word_idx_map[word]]
        embs[word] = score[1]-score[0]  
    embs[u'<unk>']=0.
    return embs
#------------------------------------------------------------------------------
class extractfeatures(object):
    def __init__(self,lexicons): 
        self.lexicons=lexicons
    def mohammad(self,tw):
        f=[]
        count=0
        scpos=0
        scneg=0
        for i,uni in enumerate(tw):
            sc=self.lexicons.get(uni,0)
            f.append(sc)
            if sc!=0:
                count+=1
            if sc>0:
                scpos+=sc
            if sc<0:
                scneg+=sc
        f=np.array(f)
        sumf=f.sum(axis=0)
        maxf=f.max(axis=0)
        lastW=self.lexicons.get(tw[-1],0)
        feature=np.array([count,sumf,maxf,lastW])
#        feature=np.array([count,scpos,maxf,lastW,scneg,sumf])
        return feature
        
    def allfeat(self,tweets,labels):
        y=[]
        x=np.array([])
        sent={u'NEG':0,u'POS':1,u'OBJ':2,u'NEUTRAL':3}
        for tw,senti in zip(tweets,labels):
            y.append(sent[senti])
            feature=self.mohammad(tw.split())
            x=np.concatenate([x,feature])
        x=x.reshape((len(y),len(x)/len(y)))
        return(x,y)
def writevec(filename,x,y):
    idx=np.array(range(len(y)))
    with cd.open(filename,'wb',encoding='utf8') as f:
        for i in idx:
            f.write(unicode(str(y[i])+'\t'))
            feature=x[i]
            for (j,k) in enumerate(feature):
                f.write(unicode(str(j+1)+':'+str(k)+' '))
            f.write(unicode('\n'))
    
def main(lexicons,model,datatype):
    features=extractfeatures(lexicons)
    gr = AraTweet()
    if datatype=="balanced":
        dataset=dict(name="4-balanced", params=dict(klass="4", balanced="balanced"))
    else:
        dataset=dict(name="4-unbalanced", params=dict(klass="4", balanced="unbalanced"))
    (d_train, l_train, d_test, l_test, d_valid, l_valid) = gr.get_train_test_validation(**dataset['params'])
    d_train = np.concatenate((d_train, d_valid))
    l_train = np.concatenate((l_train, l_valid))
    
    print "extracting features for training"
    x_train,y_train=features.allfeat(d_train,l_train)
    fname='../data/libdata/astd/libformat/train'+model+'.'+datatype
    writevec(fname,x_train,y_train)
    
    print "extracting features for testing"
    x_test,y_test=features.allfeat(d_test,l_test)
    fname='../data/libdata/astd/libformat/test'+model+'.'+datatype
    writevec(fname,x_test,y_test)
    
def selectlexicons(lexiconfile,model):
    if model.split('.')[1]=='sowe':    
        lexicons=load_sowe(lexiconfile,'../data/lexicons/mylexicon-ar/info_ar.tw')
    if model.split('.')[1]=='so':    
        lexicons=load_so(lexiconfile)
    return lexicons
   
if __name__ == "__main__":
    """
    python model-1d-astd.py  balanced ../data/lexicons/mylexicon-ar/lexicon-2d-fixW-ar/epoch_5Words_current.npy .sowe
    """
    datatype=sys.argv[1]
    lexiconfile=sys.argv[2]
    model=sys.argv[3]
    lexicons=selectlexicons(lexiconfile,model)
    main(lexicons,model,datatype)
            