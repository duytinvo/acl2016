# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:28:45 2015

@author: duytinvo
"""
import subprocess
import numpy as np
import sys

def goldfile(fname,gfile):
    with open(gfile,'wb') as g:
        with open(fname,'rb') as f:
            for line in f:
                l=line.strip().split()[0]
                g.write(l)
                g.write('\n')

def fmeasure(predfile,goldfile):
    from sklearn.metrics import precision_recall_fscore_support
    y_gold=np.loadtxt(goldfile)
    y_pred=np.loadtxt(predfile)
    #r1=precision_recall_fscore_support(y_gold, y_pred)
    r2=precision_recall_fscore_support(y_gold, y_pred, average='macro')
    #r3=precision_recall_fscore_support(y_gold, y_pred, average='micro')
    return r2


def predict(ci,trfile,tfile,pfile):
    model='./model/'+trfile.split('/')[-1]+'.model'
    traincmd=["./liblinear/train", "-c", "0.001", trfile,model]
    traincmd[2]=ci
    subprocess.call(traincmd)
    predcmd=["./liblinear/predict", tfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    preddev=float(output.split()[2].strip('%'))
    print "Predict: Learning liblinear with c=%s: %f"%(ci,preddev)
    return output
    
def CV(ci,trfile):
    model='./model/'+trfile.split('/')[-1]+'.model'
    traincmd=["./liblinear/train", "-c", "0.001", "-v", "5", trfile,model]
    traincmd[2]=ci
    p = subprocess.Popen(traincmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    preddev=float(output.split()[-1].strip('%'))
    print "CV: Learning liblinear with c=%s: %f"%(ci,preddev)
    return preddev

def DEV(ci,trfile,devfile,pfile='./liblinear/preddev'):
    model='./model/'+trfile.split('/')[-1]+'.model'
    traincmd=["./liblinear/train", "-c", "0.001", trfile,model]
    traincmd[2]=ci
    subprocess.call(traincmd)
    predcmd=["./liblinear/predict", devfile, model, pfile]
    p = subprocess.Popen(predcmd, stdout=subprocess.PIPE)
    output, err = p.communicate()
    preddev=float(output.split()[2].strip('%'))
    print "DEV: Learning liblinear with c=%s: %f"%(ci,preddev)
    return preddev    
    
def frange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r *= step
def tuneC(trfile,devfile):
    c=[]
    crange=frange(0.00001,1,10)
    c.extend([i for i in crange])
    crange=frange(0.00003,3,10)
    c.extend([i for i in crange])
    crange=frange(0.00005,5,10)
    c.extend([i for i in crange])
    crange=frange(0.00007,7,10)
    c.extend([i for i in crange])
    crange=frange(0.00009,10,10)
    c.extend([i for i in crange])
    c.sort()
    tunec=[]
    for ci in c:
#        tunec.append([ci,CV(str(ci),trfile)])
        tunec.append([ci,DEV(str(ci),trfile,devfile)])
    tunec=sorted(tunec,key=lambda x: x[1],reverse=True)
    return tunec
    
def main(trfile,devfile,tfile,pfile,rfile):
    tunec=tuneC(trfile,devfile)
    bestc=tunec[0][0]
    bestCV=tunec[0][-1]
    test=predict(str(tunec[0][0]),trfile,tfile,pfile)
    bestAcc=test  
    l1="[Tuning]: Testing on %s, the best accuracy is %f at c=%f \n"%(devfile,bestCV,bestc)
    l2="[Testing]: Testing on %s, %s"%(tfile,bestAcc)
    with open(rfile,'wb') as f:
        f.write(l1)
        f.write(l2)
if __name__ == "__main__":
    """
        python runliblinear.py ../data/libdata/semeval/libformat/train.expand.ed ../data/libdata/semeval/libformat/dev.expand.ed ../data/libdata/semeval/libformat/test.expand.ed ../data/libdata/semeval/predict/expand.ed> ../data/libdata/semeval/result/expand.ed
    """
    trfile=sys.argv[1]
    devfive=sys.argv[2]
    tfile=sys.argv[3]
    pfile=sys.argv[4]
    rfile=sys.argv[5]
    main(trfile,devfive,tfile,pfile,rfile)
    
