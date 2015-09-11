__author__ = 'ymo'
import pycrfsuite
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from numpy import array
def sequence_uncertainty(tagger, xseq):
    use_marginal = True
    lseq = tagger.tag(xseq)
    if use_marginal:
        mseq = [ tagger.marginal(lseq[i], i) for i in range(len(lseq))]
        useq = [ min(p , 1.0-p) for p in mseq]
        return max(useq)
    else:
        p = tagger.probability(lseq)
        return min(p, 1.0-p)

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
    return trainer

def traincrf(feats, labels, tag, crf=getTrainer()):
    for xseq, yseq in zip(feats, labels):
        crf.append(xseq, yseq)
    crf.train('model/%s.crf' % tag)

def predictcrf(feats, tag):
    tagger = pycrfsuite.Tagger()
    tagger.open('model/%s.crf' % tag)
    y_pred = []
    y_score = []
    for xseq in feats:
        lseq = tagger.tag(xseq)
        mags = []
        for i in range(len(lseq)):
            mag = tagger.marginal(lseq[i], i)
            if lseq[i] == 'token':
                mag = 1.0 - mag
            mags.append(mag)

        y_pred.append(lseq)
        y_score.append(mags)
    return y_pred, y_score

def bio_classification_report(y_true, y_pred):

    lb = LabelBinarizer()
    y_true_combined = 1 - lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = list(chain.from_iterable(y_pred))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    print 'True sum %d Pred sum %d Len %d' %(sum(y_true_combined), sum(y_pred_combined), len(y_pred_combined))
    print "AUC P-R: %.4f" % average_precision_score(y_true_combined, y_pred_combined, average=None)
    print "AUC ROC: %.4f" % roc_auc_score(y_true_combined, y_pred_combined, average=None)
    return classification_report(
        1 - y_true_combined,
        [0 if v > 0.1 else 1 for v in y_pred_combined],
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )