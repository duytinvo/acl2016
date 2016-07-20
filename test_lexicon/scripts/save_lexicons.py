# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 23:29:41 2016

@author: duytinvo
"""
import sys
import codecs as cd
import cPickle as pickle
import numpy as np
import operator

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

def wlexicon(wfile,lexicons):
    sorted_lexicons = sorted(lexicons.items(), key=operator.itemgetter(1),reverse=True)
    with cd.open(wfile,'wb',encoding='utf8') as f:
        for (k,v) in sorted_lexicons:
            nl=k+u'\t'+str(v) +u'\n'
            f.write(nl)
if __name__=="__main__":
    lexiconfile=sys.argv[1]
    infofile=sys.argv[2]
    wfile=sys.argv[3]
    lexicons=load_sowe(lexiconfile,infofile)
    wlexicon(wfile,lexicons)