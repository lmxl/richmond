import os, math

path = '~/Dropbox/projects/richmond/'
os.chdir(os.path.expanduser(path))

import cPickle as pickle
# import richmond
# feats, labels, fines = richmond.loadEx()
# pickle.dump((feats, labels, fines), open( "/tmp/ex.pickle", "wb" ) )

feats, labels, fines = pickle.load(open(os.path.expanduser('~/tmp/ex.pickle')))

from sklearn.cross_validation import train_test_split
feats_pool, feats_val, label_pool, label_val = \
    train_test_split(feats, fines, test_size=0.8, random_state=42)
from hybrid_learner import *

l = 1.0
e = 0.1
ratio_p = 0.5
cost_r = 5
budget_t = 50
target_prev = 0
exp = [[], [], feats_val, feats_pool, [], [], label_val, label_pool]
for i in range(100):
    if random.random() > ratio_p:
        heads = True
    else:
        heads = False
    if heads:
        exp2, obj = hybrid_learner(exp, active=True, initial_size=400, fine_ratio=0.01, step_size=budget_t)
    else:
        exp2, obj = hybrid_learner(exp, active=True, initial_size=400, fine_ratio=0.99, step_size=budget_t/cost_r)

    target = obj['pr']
    if i == 0:
        target_prev = target

    print 'UPDATE', ratio_p, heads, target, target_prev
    obj['ratio_p'] = ratio_p
    obj['heads'] = heads
    if heads:
        ratio_p = min(ratio_p*l*math.exp(target-target_prev), 1 - e)
    else:
        ratio_p = max(1-(1-ratio_p)*l*math.exp(target-target_prev), e)
    obj['ratio_p_next'] = ratio_p
    target_prev = target
    exp = exp2
    trace(obj)