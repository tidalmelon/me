# -*- coding: utf-8 -*-
import sys
import os
import math

TrainingDataFile = 'newdata.train'
ModelFile = 'newdata.model'
TestDataFile = 'newdata.test'
TestOutFile = 'newdata.out'
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
        #({wid:1}, classid)
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
        # wid:({cid:sum, cid:sum}, {classid: 0.0, classid: 0.0})
        # wid 在某个分类下的出现次数和（从全部docments统计)
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
                # 第二轮迭代，由于新“信息”的增加，熵会变化
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
                    # wid:({cid:sum, cid:sum}, {classid: 0.0})
                    #({wid:1}, classid)
                    FeaClassTable[wid][1][classid] += classProbs[i] * doc[0][wid]
        #update feature weights
        for wid in WordDic.keys():
            for classid in ClassList:
                empValue = 0.0
                # wid:({cid:sum, cid:sum}, {classid: 0.0})
                if classid in FeaClassTable[wid][0]:
                    empValue = FeaClassTable[wid][0][classid]
                modelValue = 0.0
                if classid in FeaClassTable[wid][1]:
                    modelValue = FeaClassTable[wid][1][classid]
                if empValue == 0.0 or modelValue == 0.0:
                    continue
                # wid : {classid: 0.0}
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
    for classidlist = in FeaWeigths.values():
        for classid in classidlist:
            ClassList.append(classid)
    print len(FeaWeigths), ' words!', len(ClassList), ' class!'


def Predict(doc):
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

