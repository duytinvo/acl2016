# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 21:21:18 2015

@author: duytinvo
"""
import sys,re
from collections import defaultdict
import codecs as cd
from twtokenize import tokenizeRawTweetText as tokenize
from twtokenize import url,AtMention
import cPickle as pickle
import numpy as np

usernorm=re.compile(unicode(AtMention.decode('utf-8')), re.UNICODE)
urlnorm=re.compile(unicode(url.decode('utf-8')), re.UNICODE)

#-----------------------------------------------------------------------------#   
#           Stream each line of training data (format: label tweet)           # 
#-----------------------------------------------------------------------------# 
class streamtw(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        with open(self.fname,'rb') as f:
            for line in f:
                parts = line.strip().lower().split()
                if len(parts)>=2:
                    y = parts[0]
                    tw=' '.join(parts[1:])
                    x = tokenize(tw.decode('latin1')) # decoding in Latin1 for Arabic
                    x=u' '.join(x)
                    yield x,y
#-----------------------------------------------------------------------------#   
# Process training data (tokenize, substitute username, http, emoticons,...)  # 
#-----------------------------------------------------------------------------#                     
class Vocab:        
    @staticmethod
    def create_vocabulary(fname,sentences,cutoff):
        vocab = defaultdict(float)
        max_l=-np.inf
        c=0
        with cd.open(fname,'wb',encoding='utf-8') as f:
            for sent in sentences:
                line,label = sent 
                words = Vocab.process_line(line)
                for word in words:
                    vocab[word] += 1
                if max_l<=len(words):
                    max_l=len(words)
                nline=label+u' '+u' '.join(words)+u'\n'
                f.write(nline)
                c+=1
                if c%10000==0:
                    print "Vocab progress: read %d tweets" % (c)
        lst = [u"<unk>"]+[ x for x, y in vocab.iteritems() if y > cutoff ]
        vocabx = dict([ (y,x) for x,y in enumerate(lst) ])
        info={}
        info['vocab']=vocabx
        info['max_l']=max_l
        info['nosent']=c
        print "Corpus has %d unique words, %d sentences with max length %d" % (len(vocabx),c,max_l)
        return info

    @staticmethod
    def process_line(line):
        line=re.sub(url,u'<url>',line)
        line=re.sub(AtMention,u'<user>',line)
        pos=[u':)',u': )',u':-)',u':D',u'=)']
        neg=[u':(',u': (',u':-(']
        emoti=pos+neg
        for emo in emoti:
            line=line.replace(emo,u'')
        return line.split()
        
    @staticmethod
    def save(fname,vocab):
        with cd.open(fname, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        print  "Finish saving the vocabulary file"

if __name__=="__main__":
    """
    This script is used for processing emoticon-based tweets by tokenizing and replacing special tokens
    - Input:
        + Raw data with labels (format: label(1 or 0) tweet)
    - Output:
        + info file
        + process file
    ex:
    python process.py ../data/alexgo/raw/metatweets ../data/alexgo/processed/info.tw ../data/alexgo/processed/process.tw
    """
    metafile=sys.argv[1]
    infofile=sys.argv[2]
    processfile=sys.argv[3]
    sentences=streamtw(metafile)
    info=Vocab.create_vocabulary(processfile,sentences,5)
    Vocab.save(infofile,info)



