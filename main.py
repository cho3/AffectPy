'''
main.py: load, parse, classify data, score according to dataset aware metrics
'''
import sys
import xml.etree.ElementTree as ET # parse xml file
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy import sparse
from math import sqrt
from nltk.corpus import reuters


def eval_model(y_true, y_est, m_train=0):

    # number of labelled data elements used
    m = len(y_true) + m_train
    n = len(y_true)
    #m = y_true.shape[0] + m_train 

    # r2_score for regression

    # classification performance:
    acc = accuracy_score(y_true, y_est)

    # std
    std = sqrt(acc * (1. - acc) / n )

    ub, lb = acc + 1.96*std, acc - 1.96*std

    print "Acc: ({0}, {1}), labeled: {2}".format(lb, ub, m)

    return lb, ub, m

# load and handle data inputs
class Loader:

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.affect2ind = {"anger":0,       # plutchik/eckman
                            "disgust":1,
                            "fear":2,
                            "joy":3,
                            "sadness":4,
                            "surprise":5,
                            "anticipation":6, # unique to plutchik
                            "trust":7,
                            "negative":8,       # valence
                            "positive":8}
        self.punctuation = '.,\'"!?:; \n\t\r'
        self.punctset = set([c for c in self.punctuation])
        self.stopwords = set()
        with open("stopwords_en.txt") as stops:
            for line in stops:
                self.stopwords.add(line.strip())

    def load_semeval(self, fname="semeval2007/AffectiveText.test/affectivetext_test.", nb_train=0):

        data = []
        uid2affect = {}

        f2 = fname + "emotions.gold"

        with open(f2) as score_file:
            for line in score_file:
                datum = [int(dat.strip()) for dat in line.split()]
                uid = datum[0]
                affects = datum[1:]
                uid2affect[uid] = affects
        # ...
        fname0 = fname + "xml"
        tree = ET.parse(fname0)
        for i,child in enumerate(tree.getroot()):
            # TODO text norm
            text = ' '.join([w.lower().strip(self.punctuation) for w in child.text.split()])
            uid = int(child.attrib["id"])
            data.append((text, uid2affect[uid],))
            if i < nb_train:
                for word in text:
                    if word in self.stopwords:
                        continue
                    if not (word in self.word2id):
                        n = len(self.word2id)
                        self.word2if[word] = n

        return data
        # ret: list of (string, [int class])


    def load_lexicon(self, fname="NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"):

        word2affect = {}
        # ...
        with open(fname) as in_file:
            header = 0
            for line in in_file:
                if header == 2:
                    datum = line.strip().split('\t')
                    if len(datum) < 3:
                        print datum
                        continue
                    if not datum[2]:
                        print datum
                        continue
                    word, affect, val = [dat.strip(' \t\n') for dat in datum]
                    val = int(val)

                    idx = self.affect2ind.get(affect,-1)
                    # something went wrong
                    if idx == -1:
                        print line, label, datum
                        break
                    affects = word2affect.get(word, [0]*9)
                    affects[idx] = val
                    word2affect[word] = affects
                       
                elif header > 2:
                    print "wtf just happened"
                    return
                if "....." in line:
                    header += 1

        data = []
        for word, affects in word2affect.iteritems():
            data.append((word, affects,))
            if not (word in self.word2id):
                n = len(self.word2id)
                self.word2id[word] = n

        return data
        # ret: list of (string, [int class])

    def load_reuters(self):

        sents = reuters.sents()
        print "Done loading reuters, cleaning..."
        # clean, etc...
        data = []
        for sentence in sents:
            x = []
            for word in sentence:
                if word in self.punctset:
                    continue
                w = word.lower().strip(self.punctuation)
                ind = self.word2id.get(w, -1)
                if ind < 0:
                    ind = len(self.word2id)
                    self.word2id[w] = ind
                x.append(ind)
            data.append(x)
        print "Done cleaning reuters, vectorizing..."

        X = sparse.lil_matrix( (len(self.word2id), len(data),) )

        for (j, dat) in enumerate(data):
            for i in dat:
                X[i,j] = 1

        return X
        # ret: np.sparse

    def to_bow(self, X):
        # param X: [(string, [int],)]
        # convert to vectorized representation
        
        # ...
        BOW = sparse.lil_matrix((len(self.word2id), len(X),))
        Y = np.zeros((6, len(X),))

        for i, (line, affects) in enumerate(X):
            for j, aff in enumerate(affects):
                if aff <= 50:
                    continue
                Y[j,i] = 1 # aff
            #Y[i, :] = affects

            words = [w.lower().strip() for w in line.strip().split()]

            for word in words:
                idx = self.word2id.get(word, -1)
                if idx < 0:
                    continue
                BOW[idx,i] += 1

        return BOW, Y
        # ret: [[int]], [[int]]

    def build_affect_matrix(self, X):
        # param X: [(string, [int])]

        mat = sparse.lil_matrix((6, len(X)))

        for i, (word, affects) in enumerate(X):
            for j, aff in enumerate(affects):
                if j >= 6:
                    continue
                if aff != 0:
                    mat[j,i] = aff
        return mat


def main(args):


    # load dataset
    ld = Loader()

    X_gold = ld.load_semeval()
    X_lex = ld.load_lexicon()

    '''
    simple semi-supervised approach
        train a LSI model on reuters dataset, use labelled subsample to generate kNN regions in
        LSI space
    '''

    Z = ld.load_reuters()

    u, s, v = sparse.linalg.svds(Z, k=30)

    print "done computing latent space"
    '''
    # determine "energy" of each singular value

    _s = s / s.sum()
    csum = 0
    for i, _z in enumerate(_s):
        csum += _z
        print "{0}: {1}".format(i, csum)
    '''
    # inspection suggests k=30 is best
    '''
        k=30 is fairly high dimensional-- 3 options:
            normal L2 distance
            cosine distance (popular)
            fractional norms (Lk, k < 1) NOTE: violates triangle inequality--can't use w/ ball tree
        or hit it with another (nonlinear) classifier--e.g. decision tree
    '''


    # convert 'gold' set to bag of words representation
    X, Y = ld.to_bow(X_gold)

    print "done converting to BoW"
    Xlsi = np.dot(u.transpose(), X)

    print Xlsi.shape

    _Xlsi = Xlis.getA() # make dense for slicing

    for affect, i in ld.affect2ind.iteritems():
        if i >= 6:
            continue

        clf = DecisionTreeClassifier()

        #clf = KNeighborsClassifier(n=3)


        clf.fit(_Xlsi[:,0:100],Y[i,0:100].tranpose.flatten())

        y_true = Y[i,100:].transpose().flatten()
        _y_pred = clf.predict(_Xlis[:,100:])

        print affect
        eval_model(y_true, _y_pred, 100)
    

    ######

    '''
    Lexicon based approach to affect prediction
        The lexicon is a standin for feature engineering,
        And the labelled dataset is only used for model evaluation
    '''
    # convert lexicon to a sparse word-affect binary association matrix
    W_lex = ld.build_affect_matrix(X_lex)

    # this gives approximately the counts of words in each headline that has a particular affect association
    _Y = np.dot(W_lex, X)

    _Y = _Y.todense()

    # normalize results to estimate the 'affect-ness' of each sentence
    w = X.sum(axis=0) # column wise sum
    w = np.maximum(w, np.ones(w.shape))

    for affect, i in ld.affect2ind.iteritems():
        if i >= 6:
            continue

        # misc slicing and formatting to make indexing work out for evaluation funcitons
        y = _Y[i,:]
        y = np.divide(y, w) # normalize

        Y_pred = np.round(y)

        Y_true = Y[i,:]

        Y_pred = [y for y in Y_pred.getA1()]
        Y_true = [y for y in Y_true]

        print affect

        eval_model(Y_true,Y_pred)

    '''
        Lexicon based model has ~90% accuracy for classification. 
        The standard deviation of this estimate is ~30%. Thus to get a ~ +/- 1% accurate estimate 
            need approximately 3600 samples. 
    '''

    #####

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
