__author__ = 'ymo'
from base_learner import *
from ex_util import *
from sklearn.cross_validation import train_test_split
from coarse_learner import *
from numpy import array
from random import sample
from itertools import chain


def trainmulticrf(feats_train_fine, feats_train_coarse, label_train_fine, label_train_coarse, tag):
    fine_tags = list(set(chain(*label_train_fine)))
    traincrf(feats_train_fine + feats_train_coarse, masklabel(label_train_fine + label_train_coarse), tag + '__coarse')
    for fine_tag in fine_tags:
        traincrf(feats_train_fine, masklabel(label_train_fine, mask=fine_tag), tag+'_' + fine_tag)
    return fine_tags

def predictmulticrf(feats_val, fine_tags, tag):
    return


def hybrid_learner(exp, step_size=50, initial_size=200, active = False, fine_ratio = 1.0):
    step_size_fine = int(step_size*fine_ratio)
    step_size_coarse = step_size - step_size_fine
    feats_train_coarse, feats_train_fine , feats_val, feats_pool, label_train_coarse,\
        label_train_fine, label_val, label_pool = exp
    if len(feats_train_fine) + len(feats_train_coarse) < initial_size:
        feats_train_fine, feats_pool, label_train_fine, label_pool = \
            train_test_split(feats_pool, label_pool, train_size=initial_size, random_state=42)
        if fine_ratio < 1.0:
            feats_train_fine, feats_train_coarse, label_train_fine, label_train_coarse = \
                train_test_split(feats_train_fine, label_train_fine,
                             train_size=int(fine_ratio*initial_size), random_state=42)
    if active:
        tag = 'hybrid_active_%d_%d' % (len(label_train_fine), len(label_train_coarse))
    else:
        tag = 'hybrid_passive_%d_%d' % (len(label_train_fine), len(label_train_coarse))
    trainmulticrf(feats_train_fine, feats_train_coarse, label_train_fine, label_train_coarse, tag)
    return [ feats_train_coarse, feats_train_fine , feats_val, feats_pool, label_train_coarse,\
             label_train_fine, label_val, label_pool ]
