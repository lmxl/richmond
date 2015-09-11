from base_learner import *
from ex_util import *
from sklearn.cross_validation import train_test_split
from numpy import array
from random import sample
def coarse_uncertainty(feats, tag):
    tagger = pycrfsuite.Tagger()
    tagger.open('model/%s.crf' % tag)
    y_unk = [sequence_uncertainty(tagger, xseq) for xseq in feats]
    return y_unk

def coarse_learner(exp, step_size=50, initial_size=200, active = False):
    feats_train, feats_val, feats_pool, label_train, label_val, label_pool = exp
    if len(feats_train) < initial_size:
        feats_train, feats_pool, label_train, label_pool = \
            train_test_split(feats_pool, label_pool, train_size=initial_size, random_state=42)
    if active:
        tag = 'coarse_active_%d' % len(label_train)
    else:
        tag = 'coarse_passive_%d' % len(label_train)
    print '= Experiment on [%s]' % tag
    print '<-',len(feats_train), len(feats_pool),len(label_train), len(label_pool)
    traincrf(feats_train, masklabel(label_train), tag)
    pred_val, score_val = predictcrf(feats_val, tag)
    open('report/%s_report.txt' % tag,'w').write(tag + '\r\n' + bio_classification_report(masklabel(label_val), score_val))
    if active:
        unk = array(coarse_uncertainty(feats_pool, tag))
        #print unk
        #print sum(unk)/len(label_pool)
        indices_next = unk.argsort()[-step_size:][::-1]
        #print sum(unk[indices_next])/step_size
    else:
        indices_next = sample(range(len(label_pool)),step_size)
    feats_next, feats_rest, labels_next, labels_rest = partition_set(feats_pool, label_pool, indices_next)
    feats_train.extend(feats_next)
    label_train.extend(labels_next)
    feats_pool=feats_rest
    label_pool=labels_rest
    print '->',len(feats_train), len(feats_pool),len(label_train), len(label_pool)
    return [feats_train, feats_val, feats_pool, label_train, label_val, label_pool]