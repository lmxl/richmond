__author__ = 'ymo'

import sys

sys.path.append("../stone/")
from os import listdir
from os.path import exists
from getFiles import fetch_file
from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split
import pycrfsuite
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from multiprocessing import Pool

from nltk.tokenize import wordpunct_tokenize as tokenize

def getTrainer():
    crfpar={
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 40,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    }
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params(crfpar)
    return crfpar

def traincrf(feats, labels, tag, crf=getTrainer()):
    for xseq, yseq in zip(feats, labels):
        crf.append(xseq, yseq)
    crf.train('model/%s.crf' % tag)

def predictcrf(feats, tag):
    tagger = pycrfsuite.Tagger()
    tagger.open('model/%s.crf' % tag)
    y_pred = [tagger.tag(xseq) for xseq in feats]
    return y_pred

def siblings(tag):
    words = []
    coarse = []
    fine = []
    tag = tag.next_sibling
    while tag is not None and tag.name != 'milestone':
        comma = False
        if tag.name is not None:
            text = tag.text
            co = tag.name
            if tag.name in ['orgname']:
                co = tag.name
                if 'type' in tag.attrs:
                    fi = tag.attrs['type']
                else:
                    fi = 'undefined'
            else:
                co = 'token'
                fi = 'token'
        else:
            text = tag.string
            co = 'token'
            fi = 'token'
        tokens = tokenize(text)
        for seg in tokens:
            if len(seg.strip()) == 0:
                continue
            words.append(seg)
            coarse.append(co.lower())
            fine.append(fi.lower())
            # print '%s(%s) '%(seg, co)
        tag = tag.next_sibling
    return words, coarse, fine


def processfile(filepath):
    filefeatures = []
    filelabels = []
    filesents = []
    filefines = []
    raw = open(filepath).read().decode(encoding='ascii', errors='ignore')
    summary_html = BeautifulSoup(raw)
    for tag in summary_html.findAll(unit="sentence")[11:]:
        sent, label, fine = siblings(tag)
        if len(label) < 6:
            continue
        filesents.append(sent)
        filefeatures.append(sent2features(sent))
        filelabels.append(label)
        filefines.append(fine)
    return filesents, filefeatures, filelabels, filefines


def word2features(sent, i):
    word = sent[i]
    features = [
        'bias',
        #'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.len=%d' % len(word)
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        features.extend([
          #  '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper()
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper()
        ])
    else:
        features.append('EOS')
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    print "pr  AUC: %.4f" % average_precision_score(y_true_combined, y_pred_combined)
    print "roc AUC: %.4f" % roc_auc_score(y_true_combined, y_pred_combined)
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )

def loadEx():
    global home
    home = fetch_file(
        "https://onedrive.live.com/download?resid=CAE73F546D5A29CD!7653&authkey=!AEvFogi87JUJcBo&ithint=file%2cjson")
    feats = []
    sents = []
    labels = []
    fines = []
    files = [home + mfile for mfile in listdir(home) if mfile.endswith('.xml')][::20]
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

def coarse_passive(exp, step_size=50, initial_size=200):
    feats_train, feats_val, feats_pool, label_train, label_val, label_pool = exp
    if len(feats_train) < initial_size:
        feats_train, feats_pool, label_train, label_pool = \
            train_test_split(feats_pool, label_pool, train_size=initial_size, random_state=42)
    tag = 'coarse_passive_%d' % len(label_train)
    print 'Experiment on [%s]' % tag
    traincrf(feats_train, label_train, tag)
    pred_val = predictcrf(feats_val, tag)
    print(bio_classification_report(label_val, pred_val))
    return [feats_train, feats_val, feats_pool, label_train, label_val, label_pool]


if __name__ == "__main__":
    feats, labels, fines = loadEx()
    feats_pool, feats_val, label_pool, label_val = \
        train_test_split(feats, fines, test_size=0.1, random_state=42)
    exp = [[], feats_val, feats_pool, [], label_val, label_pool]
    coarse_passive(exp)






