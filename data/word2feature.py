# -*- coding: utf-8 -*-
import os


WORD_ID_DIC = {}
WORD_LIST = []
CLASS_ID_DIC = {'fi':0,'lo':1,'co':2,'ho':3,'ed':4,'te':5,'ca':6,'ta':7,'sp':8,'he':9,'ar':10,'fu':11}


def Word2Wid(fname):
    with open(fname, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            line = line.decode('utf-8')
            arr = line.split()
            words = arr[1:]
            for word in words:
                if word not in WORD_ID_DIC:
                    WORD_LIST.append(word)
                    WORD_ID_DIC[word] = len(WORD_LIST)

def word2fea(fname, feaname):
    f1 = open(feaname, 'w')
    with open(fname, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            line = line.decode('utf-8')
            arr = line.split()
            features = []
            classid = CLASS_ID_DIC[arr[0]]
            features.append(str(classid))
            words = arr[1:]
            for word in words:
                wid = WORD_ID_DIC[word]
                features.append(str(wid))
            newline = '\t'.join(features)
            f1.write(newline + os.linesep)

    f1.close()




trainname = 'train.dat'
Word2Wid(trainname)
testname = 'test.dat'
Word2Wid(testname)

word2fea(trainname, 'newdata.train')
word2fea(testname, 'newdata.test')







