# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:07:56 2015

@author: duytinvo
"""

import cPickle as pickle
import numpy as np
import codecs as cd
import os
import sys   

class streamtw(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        with cd.open(self.fname,'rb',encoding='utf-8') as f:
            for line in f:
                parts = line.strip().lower().split()
                y = parts[0]
                x = parts[1:]
                yield x,int(y)
      
def load_sspe(dirname):
    sspe={}
    fnames=os.listdir(dirname)
    for fname in fnames:
        rfile=os.path.join(dirname,fname)
        with open(rfile,'rb') as f:
            for line in f:
                parts=line.strip().split('\t')
                sspe['\t'.join(parts[0:-1])]=float(parts[-1])
    sspe['<unk>']=0.
    return sspe
         
def load_expand(rfile):
    expand={}
    fl=True
    with open(rfile,'rb') as f:
        for line in f:
            if fl:
                fl=False
                continue
            parts=line.strip().split('\t')
            tk='-'.join(parts[0].split('-')[1:])
            expand[tk]=float(parts[5])-float(parts[3])
    expand['<unk>']=0.
    return expand
     
def load_so(rfile):
    so={}
    with open(rfile,'rb') as f:
        for line in f:
            parts=line.strip().split('\t')
            so[parts[0]]=float(parts[1])
    so['<unk>']=0.
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
    embs['<unk>']=0.
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
        feature=np.array([count,scpos,maxf,lastW,scneg,sumf])
        return feature
    def allfeat(self,dataf):
        y=[]
        x=np.array([])
        stream=streamtw(dataf)
        for tw,senti in stream:
            y.append(senti)
            feature=self.mohammad(tw)
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
    
def main(lexicons,trfile,devfile,tfile,model):
    features=extractfeatures(lexicons)
    print "extracting features for training"
    x_train,y_train=features.allfeat(trfile)
    ntrfile=os.path.split(trfile)[-1]+model
    fname=os.path.join('../data/libdata/semeval/libformat/',ntrfile)
    writevec(fname,x_train,y_train)
    
    print "extracting features for developing"
    x_dev,y_dev=features.allfeat(devfile)
    ndevfile=os.path.split(devfile)[-1]+model
    fname=os.path.join('../data/libdata/semeval/libformat/',ndevfile)
    writevec(fname,x_dev,y_dev)
    
    print "extracting features for testing"
    x_test,y_test=features.allfeat(tfile)
    ntfile=os.path.split(tfile)[-1]+model
    fname=os.path.join('../data/libdata/semeval/libformat/',ntfile)
    writevec(fname,x_test,y_test) 
    
def selectlexicons(lexiconfile,model):
    if model.split('.')[1]=='sspe':
        lexicons=load_sspe(lexiconfile)
    if model.split('.')[1]=='expand':    
        lexicons=load_expand(lexiconfile)
    if model.split('.')[1]=='so':    
        lexicons=load_so(lexiconfile)
    if model.split('.')[1]=='sowe':    
        lexicons=load_sowe(lexiconfile,'../data/lexicons/mylexicon/info.tw')
    return lexicons
if __name__ == "__main__":
    """
    python features.py ../data/supervised/train ../data/supervised/dev ../data/supervised/test ../data/lexicons/mylexicon/lexicon-english/epoch_5Words_current.npy .sowe
    """ 
    trfile=sys.argv[1]
    devfile=sys.argv[2]
    tfile=sys.argv[3]
    lexiconfile=sys.argv[4]
    model=sys.argv[5]
    lexicons=selectlexicons(lexiconfile,model)
    main(lexicons,trfile,devfile,tfile,model)
            