__author__ = 'ymo'

import sys
from coarse_learner import *
from xml_parser import *
sys.path.append("../stone/")
from os import listdir
from getFiles import fetch_file
from sklearn.cross_validation import train_test_split
from multiprocessing import Pool
from hybrid_learner import *
def loadEx():
    global home
    home = fetch_file(
        "https://onedrive.live.com/download?resid=CAE73F546D5A29CD!7653&authkey=!AEvFogi87JUJcBo&ithint=file%2cjson")
    feats = []
    sents = []
    labels = []
    fines = []
    files = [home + mfile for mfile in listdir(home) if mfile.endswith('.xml')]#[::50]
    pool = Pool(processes=11)
    it = pool.imap(processfile, files)
    for counter in range(1, len(files) + 1):
        sents1, feats1, labels1, fines1 = it.next()
        feats.extend(feats1)
        sents.extend(sents1)
        labels.extend(labels1)
        fines.extend(fines1)
    print ('Loaded training examples.')
    return feats, labels, fines


def mymain():
    feats, labels, fines = loadEx()
    feats_pool, feats_val, label_pool, label_val = \
        train_test_split(feats, fines, test_size=0.8, random_state=42)
    mexp = [
        [[], feats_val, feats_pool, [], label_val, label_pool],
        [[], feats_val, feats_pool, [], label_val, label_pool],
        [[], [], feats_val, feats_pool, [], [], label_val, label_pool],
        [[], [], feats_val, feats_pool, [], [], label_val, label_pool]
    ]
    for i in range(10):
        mexp[0] = coarse_learner(mexp[0], active=True)
        mexp[1] = coarse_learner(mexp[1], active=False)
        #mexp[2] = hybrid_learner(mexp[2], active=True)

if __name__ == "__main__":
    mymain()






