# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:36:22 2015
@author: duytinvo
"""
import codecs as cd
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import theano
import theano.tensor as T
import  os
import sys
import time
import subprocess

#-----------------------------------------------------------------------------#   
#-----------------------------------------------------------------------------# 
#-----------------------------------------------------------------------------#  
class LogisticFixW(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)
    
    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie
    
    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie
    
    """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.array([[1,0],[0,1]], dtype=theano.config.floatX),
                    name='W_LogisticRegression')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
#        if b is None:
#            self.b = theano.shared(
#                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
#                    name='b_LogisticRegression')
#        else:
#            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
#        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

    .. math::
    
    \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
    \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
    \ell (\theta=\{W,b\}, \mathcal{D})
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    
    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
#        return T.sum(self.p_y_given_x-y)
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch
    
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
#-----------------------------------------------------------------------------#   
#-----------------------------------------------------------------------------# 
#-----------------------------------------------------------------------------#  
def readinfo(fname):
    with cd.open(fname, 'rb') as f:
        word_idx_map=pickle.load(f)
    return word_idx_map
#-----------------------------------------------------------------------------#     
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
#-----------------------------------------------------------------------------#                 
def add_word_lexicons(word_vecs, word_idx_map, k=1):
    """
    Uniformly initialze lexicon scores for each word in Vocabulary
    Last token stands for paddings (set to 0)
    """
    vocab_size = len(word_idx_map)
    W = np.zeros(shape=(vocab_size+1, k))  
    W[-1] = np.zeros(k)                         #for padding and should set to zeros in SOWE
    rand_w=np.random.uniform(0,1,k)
    for word in word_idx_map.keys():
        W[word_idx_map[word]]=word_vecs.get(word,rand_w)
    return W    
#-----------------------------------------------------------------------------# 
def sent2idx(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    for word in sent:
        x.append(word_idx_map.get(word,word_idx_map[u'<unk>']))
    while len(x) < max_l:
        x.append(-1)
    return x
#-----------------------------------------------------------------------------# 
def make_batch(sentences, word_idx_map, max_l=62, batch_size=20):
    """
    Transforms sentences into a 2-d matrix.
    """
    train= []
    c=0
    for sent in sentences:
        x,y=sent   
#        datum = get_idx_from_sent(x, word_idx_map, max_l, len(x))  
        datum =sent2idx(x, word_idx_map, max_l)
        datum.append(y)
        train.append(datum)
        c+=1
        if c%batch_size==0:
            batch = np.array(train,dtype="int")
            train= []
            c=0
            yield batch
#-----------------------------------------------------------------------------#             
def save(folder, params, epoch):   
    for param in params:
        if os.path.exists(os.path.join(folder, epoch+param.name +'_current.npy')):
            subprocess.call(['mv', os.path.join(folder,epoch+ param.name +'_current.npy'), os.path.join(folder,epoch+ param.name +'_last.npy')])
        np.save(os.path.join(folder, epoch+param.name +'_current.npy'), param.get_value())
#-----------------------------------------------------------------------------# 
def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
#-----------------------------------------------------------------------------#       
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 
#-----------------------------------------------------------------------------#   
#-----------------------------------------------------------------------------# 
#-----------------------------------------------------------------------------#         
def train_sowe(trainsentences,
               word_idx_map,
               max_l,
               num_trains,
               U,
               lexicon_scores=2, 
               hidden_units=[2,2], 
               shuffle_batch=True,
               n_epochs=1, 
               batch_size=1000, 
               lr_decay = 0.95,
               sqr_norm_lim=9,
               non_static=True,
               report=100,
               valid=5):
  
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): 
        os.mkdir(folder) 
    #-------------------------------------------------------------------------------------------------------------------------------#
    #define model architecture
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(lexicon_scores)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[-1,:], zero_vec_tensor))])
    #-------------------------------------------------------------------------------------------------------------------------------#
    #build model and updating functions using adadelta    
    layer0_input = Words[T.cast(x,dtype="int32")]  
    layer1_sum=layer0_input.sum(axis=1)  
    layer1_output=layer1_sum
    classifier=LogisticFixW(input=layer1_output,n_in=hidden_units[0], n_out=hidden_units[1])
    cost = classifier.negative_log_likelihood(y) 
    #define parameters of the model
    params = [Words]
    grad_updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)
    val_model = theano.function([x,y], classifier.errors(y))                   
    train_model = theano.function([x,y], cost, updates=grad_updates) 
    #-------------------------------------------------------------------------------------------------------------------------------#
    #start training over mini-batches
    print '... training'
    epoch = 0      
    cost_batch = 0  
    total_examples=0
    train_total_costs=0
    train_total_perf=0
    last_examples=0
    #------------------------------------------------------------------------
    #validate and test model   
    while (epoch<n_epochs): 
        tic = time.time()
        epoch = epoch + 1
        idx=0
        cost_batch = 0  
        total_examples=0
        train_total_costs=0
        train_total_perf=0
        last_examples=0
        grouper=make_batch(trainsentences,word_idx_map,max_l,batch_size)
        for minibatch in grouper:
            np.random.seed(3435)
            train_set = np.random.permutation(minibatch) 
            train_set_x = train_set[:,:-1] 
            train_set_y = np.asarray(train_set[:,-1],"int32")  
            cost_batch = train_model(train_set_x,train_set_y)
            set_zero(zero_vec)
            
            acc_batch = val_model(train_set_x,train_set_y)
            perf_batch = 1- np.mean(acc_batch)
            
            total_examples+=minibatch.shape[0]
            train_total_costs += float(cost_batch)
            train_total_perf += float(perf_batch)
            idx+=1
            print '[learning] epoch %i >> %2.2f%%'%(epoch,total_examples*100./num_trains),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()
            if (total_examples-last_examples)>=batch_size*report:
                last_examples=total_examples
                avg_cost = train_total_costs/float(idx)
                avg_perf = train_total_perf/float(idx)
                t=time.time()-tic
                print "[Reporting] epoch ",epoch,": training examples=",total_examples,", speed=",total_examples/t, ' '*20
                print "==========> Average loss=", avg_cost,", perfomance=",avg_perf,' '*20
                save(folder,params,'epoch_'+str(epoch))
        print '*'*70
        print "\t\tFINISH TRAINING epoch ",epoch, 'in ', time.time()-tic
        print '*'*70
    return avg_cost,avg_perf
            
if __name__=="__main__":
    traininfofile='../data/alexgo/processed/info.tw'
    trainfile='../data/alexgo/processed/process.tw'
    info=readinfo(traininfofile)
    trainsentences=streamtw(trainfile)
    word_idx_map=info['vocab']
    max_l=info['max_l']
    num_trains=info['nosent']    
    rand_vecs = {}
    lexicon_scores=2
    U= add_word_lexicons(rand_vecs, word_idx_map, lexicon_scores)
    tic = time.time()
    perf=train_sowe(   trainsentences,
                       word_idx_map,
                       max_l,
                       num_trains,
                       U,
                       lexicon_scores, 
                       hidden_units=[lexicon_scores,2], 
                       shuffle_batch=True,
                       n_epochs=5, 
                       batch_size=50, 
                       lr_decay = 0.95,
                       sqr_norm_lim=9,
                       non_static=True,
                       report=1000)
    print perf
    print time.time()-tic