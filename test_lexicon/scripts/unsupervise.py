# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:07:56 2015

@author: duytinvo
"""
import cPickle as pickle
import numpy as np
import codecs as cd
import os

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
    def unsupervised(self,tw):
        f=0
        for uni in tw:
            sc=self.lexicons.get(uni,self.lexicons['<unk>'])
            f+=sc
        if f>=0:
            l=1
        else:
            l=0
        return l
    def allfeat(self,dataf):
        g=[]
        p=[]
        stream=streamtw(dataf)
        for tw,senti in stream:
            g.append(senti)
            l=self.unsupervised(tw)
            p.append(l)
        return(g,p)
        
def fmeasure(gl,pl):
    from sklearn.metrics import precision_recall_fscore_support
    y_gold=np.array(gl)
    y_pred=np.array(pl)
    assert len(y_gold)==len(y_pred)
#    print y_gold[:10]
#    print y_pred[:10]
    prf=precision_recall_fscore_support(y_gold, y_pred, average='macro',pos_label=None)
    acc=precision_recall_fscore_support(y_gold, y_pred, average='micro',pos_label=None)
    return prf,acc[0]
  
def main(lexicons,trfile):
    features=extractfeatures(lexicons)
    print "Predict on:",trfile
    gold,pred=features.allfeat(trfile)
    perf,acc=fmeasure(gold,pred)
    print "="*80
    print "Unsupervised Evaluation:"
    print "Precision = %.1f; Recall = %.1f; F1 = %.1f; Accuracy = %.1f"%(perf[0]*100,perf[1]*100,perf[2]*100,acc*100)
    print "="*80
    
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
    import sys
    trfile=sys.argv[1]
    lexiconfile=sys.argv[2]
    model=sys.argv[3]
    lexicons=selectlexicons(lexiconfile,model)
    main(lexicons,trfile)
            