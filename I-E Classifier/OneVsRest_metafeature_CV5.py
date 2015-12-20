#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################
import argparse
import codecs
import time
import sys
import os, re
import nltk
from collections import defaultdict
from random import shuffle, randint
import numpy as np
from numpy import array, arange, zeros, hstack, argsort
import unicodedata
from scipy.sparse import csr_matrix
# sklearn imports
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation


def getDataInFormat():
	dataTuples=[]
	fileobject=codecs.open("200g.all", "r", "utf-8")
	lstOfLines = fileobject.readlines()
	for line in lstOfLines:
		tag = line.split("\t")[0]
		tweets = line.split("\t")[3]
		dataTuples.append([tag,tweets])
	
	return dataTuples

#
#####################################


def getThreeColumnDataDict(emotionLines):
    shuffle(emotionLines)
    classes=["ISTJ","ISFJ","INFJ","INTJ","ISTP","ISFP","INFP","INTP","ESTP","ESFP","ENFP","ENTP","ESTJ","ESFJ","ENFJ","ENTJ"]
    myData={pair[0]: [] for pair in emotionLines}
    for cat in classes:
            for pair in emotionLines:
                    if pair[0]==cat:
                            myData[pair[0]].append(pair[1])

    #print "Mydata------------->",myData["HAPPINESS"]
    return myData

def getDataStats(myData):
    # Print some stats:
    ##########################
    majorClass=max([len(myData[k]) for k in myData])
    totalCount=sum([len(myData[k]) for k in myData])
    print "Majority class count: ", majorClass
    print "Total data point count: ", totalCount
    print "Majority class % in train data: ", round((majorClass/float(totalCount))*100, 2), "%"
    print "*"*50, "\n"

def getLabelsAndVectors(dataTuples):
    """ 
    Input:
	dataTuples is a list of tuples
	Each tuple in the list has
		   0=label
		   1= tweet body as unicode/string
    Returns an array of labels and another array for words 
    """
    labels=[]
    vectors=[]
    ids=[]
    c=0

    for dataPoint in dataTuples:
            ids.append(c)
            c+=1
            label, vector=dataPoint[0], dataPoint[1].split()
            if(label.find('I') != -1):
                    labels.append("Introvert")
            else:
                    labels.append("Extrovert")

            vectors.append(vector)
    return ids, labels, vectors

def getSpace(vectors):
    # get the dictionary of all words in train; we call it the space as it is the space of features for bag of words
    space={}
    for dataPoint in vectors:
            words=dataPoint
            for w in words:
                    if w not in space and w != "@USER" and w != "@URL" and w != "@HASHTAG":
                            space[w]=len(space)
    #print "space------->",space["this"]
    return space

def getReducedSpace(vectors, space):
    # get the dictionary of all words in train; we call it the space as it is the space of features for bag of words
    reducedSpace=defaultdict(int)
    for dataPoint in vectors:
            words=dataPoint
            for w in words:
                    reducedSpace[w]+=1
    for w in space:
            if reducedSpace[w] < 3:
                    del reducedSpace[w]
    reducedSpace={w: reducedSpace[w] for w in reducedSpace}
    return reducedSpace

def getOneHotVectors(ids, labels, vectors, space, followerCountsList):
    oneHotVectors={}
    triples=zip(ids, labels, vectors, followerCountsList)
    vec = np.zeros((len(space)))
    #for dataPoint in vectors:
    for triple in triples:
        idd, label, dataPoint, followerCount = triple[0], triple[1], triple[2], triple[3]
        #for t in xrange(len(space)):
        # populate a one-dimensional array of zeros of shape/length= len(space)
        vec=np.zeros((len(space))) # ; second argument is domensionality of the array, which is 1
        for w in dataPoint:
            try:
                vec[space[w]]=1
            except:
                continue
        # add emotion lexicon features
        vec=addMetaFeatures(vec, space, followerCount)
        oneHotVectors[idd]=(vec, array(label))
    return oneHotVectors

def addMetaFeatures(vec, space, followerCount):
    if(followerCount > 509):
            vec[space["highFollowerCount"]] = 1
    elif(followerCount > 181):
            vec[space["mediumFollowerCount"]] = 1
    else:
            vec[space["lowFollowerCount"]] = 1
    return vec

def getOneHotVectorsAndLabels(oneHotVectorsDict):
        vectors = []
        labels = []

        for k in oneHotVectorsDict:
                vectors.append(oneHotVectorsDict[k][0])
                labels.append(oneHotVectorsDict[k][1])

        vectorsArr = array(vectors)
        labelsArr = array(labels)
        return vectorsArr, labelsArr

def loadFollowerCountsFromFile():
	infileObject=codecs.open("200meta.all", "r", "utf-8")
	listOfLines= infileObject.readlines()
	followerCountsList = []
	for line in listOfLines:
		fcStr = line.split("\t")[0]
		fc = int(fcStr.split("=")[1])
		followerCountsList.append(fc)
	return followerCountsList

emotionFeatures=["lowFollowerCount", "mediumFollowerCount", "highFollowerCount"]

def augmentSpace(space, featuresList):
    """
    Adds a list of features to the bag-of-words dictionary, we named "space".
    """
    for f in featuresList:
        if f not in space:
            space[f]=len(space) 
    return space


def main():

    dataTuples=getDataInFormat()
    print "Length of dataTuples is: ",  len(dataTuples)
    shuffle(dataTuples)
    trainTuples=dataTuples
    del dataTuples
    ids, labels, vectors= getLabelsAndVectors(trainTuples)
    del trainTuples
    followerCountsList = loadFollowerCountsFromFile()
    space=getSpace(vectors)
    reducedSpace=getReducedSpace(vectors, space)
    spaceWithMetaFeatures= augmentSpace(reducedSpace, emotionFeatures)

    print "Total # of features in your space is: ", len(space)
    print "Total # of features in your reducedSpace is: ", len(reducedSpace)
    oneHotVectors=getOneHotVectors(ids, labels, vectors,spaceWithMetaFeatures , followerCountsList)
    trainVectors, trainLabels=getOneHotVectorsAndLabels(oneHotVectors)
    del oneHotVectors
    clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear',gamma=0.1, verbose= False, probability=False))
    clf.fit(trainVectors, trainLabels)
    
    print "\nDone fitting classifier on training data...\n"
    print "\nDone fitting classifier on training data...\n"
    print "="*50, "\n"
    print "Results with 5-fold cross validation:\n"
    print "="*50, "\n"
    predicted = cross_validation.cross_val_predict(clf, trainVectors, trainLabels, cv=5)
    print "*"*20
    print "\t accuracy_score\t", metrics.accuracy_score(trainLabels, predicted)
    print "*"*20
    print "precision_score\t", metrics.precision_score(trainLabels, predicted)
    print "recall_score\t", metrics.recall_score(trainLabels, predicted)
    print "\nclassification_report:\n\n", metrics.classification_report(trainLabels, predicted)
    print "\nconfusion_matrix:\n\n", metrics.confusion_matrix(trainLabels, predicted)
    
	
if __name__ == "__main__":
    print "OneVsRest Classifier with 5 fold validation + metafeature followerCount"
    main()
