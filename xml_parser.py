__author__ = 'ymo'

from ex_util import *
from bs4 import BeautifulSoup
from nltk.tokenize import wordpunct_tokenize as tokenize

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