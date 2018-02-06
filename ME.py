# -*- coding: utf-8 -*-
import sys
import os
import math

TrainingDataFile = 'newdata.train'
ModelFile = 'newdata.model'
TestDataFile = 'newdata.test'
DocList = []
WordDic = {}
FeaClassTable = {}
FeaWeigths = {}
ClassList = []
C = 100
MaxIteration = 1000
LogLLDif = 0.1
CommonFeaId = 10000001

def Dedup(items):
    tempDic = {}
    for item in items:
        if item not in tempDic:
            tempDic[item] = True
    return tempDic.keys()


def LoadData():
    global CommonFeaId
    i = 0
    infile = file(TrainingDataFile, 'r')
    sline = infile.readline().strip().decode('utf-8')
    maxwid = 0
    while len(sline) > 0:
        words = sline.split('\t')
        if len(words) < 1:
            print 'Format error'
            break
        classid = int(words[0])
        if classid not in ClassList:
            ClassList.append(classid)

        words = words[1:]
        # remove duplicate words , binary ditribution
        words = Dedup(words)
        newDoc = {}
        for word in words:
            if len(word) < 1:
                continue
            wid = int(word)
            if wid > maxwid:
                maxwid = wid
            if wid not in WordDic:
                WordDic[wid] = 1
            if wid not in newDoc:
                newDoc[wid] = 1
        i += 1
        DocList.append((newDoc, classid))
        sline = infile.readline().strip().decode('utf-8')
    infile.close()
    print len(DocList), "instances loaded!"
    print len(ClassList), " classes!", len(WordDic), " words !"
    CommonFeaId = maxwid + 1
    print 'Max wid:', maxwid
    WordDic[CommonFeaId] = 1


def ComputeFeaEmpDistribution():
    """
    计算约束函数f到经验分布
    FeaClassTable: 词，类别的频率的二维数组
    """
    global C
    global FeaClassTable
    FeaClassTable = {}
    for wid in WordDic.keys():
        tempPair = ({}, {})
        FeaClassTable[wid] = tempPair
    maxCount = 0
    for doc in DocList:
        if len(doc[0]) > maxCount:
            maxCount = len(doc[0])
    C = maxCount + 1
    for doc in DocList:
        #({wid:1}, classid)
        doc[0][CommonFeaId] = C - len(doc[0])
        for wid in doc[0].keys():
            if doc[1] not in FeaClassTable[wid][0]:
                FeaClassTable[wid][0][doc[1]] = doc[0][wid]
            else:
                FeaClassTable[wid][0][doc[1]] += doc[0][wid]


def GIS():
    global C
    global FeaWeigths
    for wid in WordDic.keys():
        FeaWeigths[wid] = {}
        # wid : {classid: 0.0, classid: 0.0}
        for classid in ClassList:
            FeaWeigths[wid][classid] = 0.0
    n = 0
    prelogllh = -1000000.0
    logllh = -10000.0
    while logllh - prelogllh >= LogLLDif and n < MaxIteration:
        n += 1
        prelogllh = logllh
        logllh = 0.0
        print 'iteration', n
        for wid in WordDic.keys():
            for classid in ClassList:
                FeaClassTable[wid][1][classid] = 0.0

        # compute expected values of features subject to the model p(y|x)
        for doc in DocList:
            classProbs = [0.0] * len(ClassList)
            sum = 0.0
            for i in range(len(ClassList)):
                classid = ClassList[i]
                pyx = 0.0
                for wid in doc[0].keys():
                    pyx += FeaWeigths[wid][classid]
                # 第一轮迭代，pyx=0, e^0=1,则各文档x属于各个分类的概率相等
                pyx = math.exp(pyx)
                classProbs[i] = pyx
                sum += pyx
            for i in range(len(ClassList)):
                classProbs[i] = classProbs[i] / sum

            # 得到p(y|x),计算约束函数f的模型期望EP(f) = p(y|x) * f(x,y)
            for i in range(len(ClassList)):
                classid = ClassList[i]
                if classid == doc[1]:
                    logllh += math.log(classProbs[i])
                for wid in doc[0].keys():
                    FeaClassTable[wid][1][classid] += classProbs[i] * doc[0][wid]
        #update feature weights
        for wid in WordDic.keys():
            for classid in ClassList:
                empValue = 0.0
                if classid in FeaClassTable[wid][0]:
                    empValue = FeaClassTable[wid][0][classid]
                modelValue = 0.0
                if classid in FeaClassTable[wid][1]:
                    modelValue = FeaClassTable[wid][1][classid]
                if empValue == 0.0 or modelValue == 0.0:
                    continue
                FeaWeigths[wid][classid] += math.log(FeaClassTable[wid][0][classid] / FeaClassTable[wid][1][classid]) / C
        print 'loglikelihood:', logllh

def SaveModel():
    outfile = file(ModelFile, 'w')
    with open(ModelFile, 'w') as outfile:
        for wid in FeaWeigths.keys():
            outfile.write(str(wid))
            outfile.write(' ')
            for classid in FeaWeigths[wid]:
                outfile.write(str(classid) + ' ')
                outfile.write(str(FeaWeigths[wid][classid]) + ' ')
            outfile.write('\n')

def LoadModel():
    global ClassList
    global FeaWeigths
    with open(ModelFile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            arr = line.split(' ')
            wid = int(arr[0])
            FeaWeigths[wid] = {}
            i = 1
            while i < len(arr):
                classid = int(arr[i])
                i += 1
                FeaWeigths[wid][classid] = float(arr[i])
                i += 1
    ClassList = []
    for classidlist in FeaWeigths.values():
        for classid in classidlist:
            ClassList.append(classid)
        break
    print len(FeaWeigths), ' words!', len(ClassList), ' class!'


def Predict(doc):
    """
    规范化因子 对 同一个doc为固定直
    """
    classid = ClassList[0]
    maxClass = classid
    sum = 0.0
    for wid in doc.keys():
        if wid in FeaWeigths:
            sum += FeaWeigths[wid][classid]
    max = sum
    i = 1
    while i < len(ClassList):
        sum = 0.0
        for wid in doc.keys(): 
            if wid in FeaWeigths:
                sum += FeaWeigths[wid][ClassList[i]]
        if sum > max:
            max = sum
            maxClass = ClassList[i]
        i += 1
    return maxClass

def Test():
    TrueLabelList = []
    PredLabelList = []
    i = 0
    infile = file(TestDataFile, 'r')
    line = infile.readline().strip()
    scoreDic = {}
    iline = 0
    while line:
        iline += 1
        if iline % 10 == 0:
            print iline, ' iline finished!',
            words = line.split('\t')
            classid = int(words[0])
            TrueLabelList.append(classid)
            words = words[1:]
            # binary distribution
            words = Dedup(words)
            maxScore = 0.0
            newDoc = {}
            for word in words:
                if not word:
                    continue
                wid = int(word)
                if wid not in newDoc:
                    newDoc[wid] = 1
            maxClass = Predict(newDoc)
            PredLabelList.append(maxClass)
            line = infile.readline().strip()
    infile.close()
    print len(PredLabelList), len(TrueLabelList)
    return TrueLabelList, PredLabelList

def Evaluate(TrueList, PredList):
    accuracy = 0
    for i in range(len(TrueList)):
        if TrueList[i] == PredList[i]:
            accuracy += 1
    accuracy = float(accuracy) / len(TrueList)
    print 'Accuracy:', accuracy

def CalPreRec(TrueList, PredList, classid):
    correctNum = 0
    allNum = 0
    predNum = 0
    for i in range(len(TrueList)):
        if TrueList[i] == classid:
            allNum += 1
            if TrueList[i] == PredList[i]:
                correctNum += 1
        if PredList[i] == classid:
            predNum += 1

    return float(correctNum) / predNum, float(correctNum) / allNum

if __name__ == '__main__':
    if sys.argv[1] == '1':
        print 'start training:'
        TrainingDataFile = sys.argv[2]
        ModelFile = sys.argv[3]
        LoadData()
        ComputeFeaEmpDistribution()
        GIS()
        SaveModel()
    if sys.argv[1] == '0':
        print 'start testing:'
        TestDataFile = sys.argv[2]
        ModelFile = sys.argv[3]
        LoadModel()
        TList, PList = Test()
        Evaluate(TList, PList)
        print '-------------------------------------------'
        for classid in ClassList:
            pre, rec = CalPreRec(TList, PList, classid)
            print 'precision and recall for class', classid, ':', pre, rec
