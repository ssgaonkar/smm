#!/usr/bin/env python
# -*- coding: utf-8 -*-
####################################
import argparse
import json
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
        fileobject=codecs.open("tweets.txt", "r", "utf-8")
        lstOfLines = fileobject.readlines()
        dataTuples =[]
        for line in lstOfLines:
                json_obj = json.loads(line)
                for key,value in json_obj.iteritems():
                        if(key =='text'):
                                tweet = value
                                
                                if(isRetweet(tweet)==False):
                                        label = getLabel(tweet)
                                        tweet = processTweet(tweet)
                                        dataTuples.append([label,tweet])

        return dataTuples

def getLabel(tweet):
        tweet = tweet.lower()
        label=""
        if("#anxiety" in tweet or "#anxious" in tweet):
                label = "ANXIETY"
        elif("#suicidal" in tweet or "#depressed" in tweet or "#depression" in tweet):
                label = "DEPRESSION"
        else:
                label = "NO-EMOTION"
                
        return label

def processTweet(tweet):
    tmpTweet=""
    for w in tweet.split(" "):
        
        if(w.startswith("@")):
            tmpTweet+= "@USER"
        elif(w.startswith("http")):
            tmpTweet+= "@URL"
        elif(w.startswith("#")):
            tmpTweet+= "@HASHTAG"
        else:
            tmpTweet+=w
            
        tmpTweet+=" "
        
    return tmpTweet

#####################################


def getThreeColumnDataDict(emotionLines):
    shuffle(emotionLines)
    classes=["ANXIETY","DEPRESSION","NO-EMOTION"]
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

    labels=[]
    vectors=[]
    ids=[]
    c=0
    for dataPoint in dataTuples:
        ids.append(c)
        c+=1
        label, vector=dataPoint[0], dataPoint[1].split()
        
        labels.append(label)
        vectors.append(vector)

    return ids, labels, vectors

def getSpace(vectors):
    # get the dictionary of all words in train; we call it the space as it is the space of features for bag of words
    space={}
    for dataPoint in vectors:
        words=dataPoint
        for w in words:
            if w not in space and w != "@USER" and w != "@URL" and w != "@HASHTAG" and w != "/":
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

def getOneHotVectors(ids, labels, vectors, space):
    oneHotVectors={}
    triples=zip(ids, labels, vectors)
    vec = np.zeros((len(space)))
    #for dataPoint in vectors:
    for triple in triples:
        idd, label, dataPoint= triple[0], triple[1], triple[2]
        #for t in xrange(len(space)):
        # populate a one-dimensional array of zeros of shape/length= len(space)
        vec=np.zeros((len(space))) # ; second argument is domensionality of the array, which is 1
        for w in dataPoint:
            try:
                vec[space[w]]=1
            except:
                continue
        oneHotVectors[idd]=(vec, array(label))
    #print "oneHotVectors-------------->",idd,oneHotVectors[idd]
    return oneHotVectors

def getOneHotVectorsAndLabels(oneHotVectorsDict):
    vectors= array([oneHotVectorsDict[k][0] for k in oneHotVectorsDict])
    labels= array([oneHotVectorsDict[k][1] for k in oneHotVectorsDict])
    print "labels.shape", labels.shape 
    print "vectors.shape", vectors.shape
    #print "labels-------->",len(labels),"------",labels
    return vectors, labels


def isRetweet(tweet):
    if tweet.lower().split()[0] =="rt":
        return True
    return False

def main():

    dataTuples=getDataInFormat()
    print "Length of dataTuples is: ",  len(dataTuples)
    shuffle(dataTuples)

    trainTuples=dataTuples
    myData=getThreeColumnDataDict(dataTuples)
    totalCount=sum([len(myData[k]) for k in myData])
    getDataStats(myData)
    ids, labels, vectors= getLabelsAndVectors(trainTuples)
    space=getSpace(vectors)
    reducedSpace=getReducedSpace(vectors, space)
    print "Total # of features in your space is: ", len(space)
    print "Total # of features in your reducedSpace is: ", len(reducedSpace)
    oneHotVectors=getOneHotVectors(ids, labels, vectors, reducedSpace)
    vectors, labels=getOneHotVectorsAndLabels(oneHotVectors)
    del oneHotVectors
    trainVectors = vectors
    trainLabels = labels
    print trainLabels
    del vectors
    del labels
    clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear',gamma=0.1, verbose= False, probability=False))
    clf.fit(trainVectors, trainLabels)
    
    print "\nDone fitting classifier on training data...\n"
    print "="*50, "\n"
    print "Results with 10-fold cross validation:\n"
    print "="*50, "\n"
    
    predicted = cross_validation.cross_val_predict(clf, trainVectors, trainLabels, cv=10)
    print "*"*20
    print "\t accuracy_score\t", metrics.accuracy_score(trainLabels, predicted)
    print "*"*20
    print "precision_score\t", metrics.precision_score(trainLabels, predicted)
    print "recall_score\t", metrics.recall_score(trainLabels, predicted)
    print "\nclassification_report:\n\n", metrics.classification_report(trainLabels, predicted)
    print "\nconfusion_matrix:\n\n", metrics.confusion_matrix(trainLabels, predicted)


if __name__ == "__main__":
    print "Emotion  Disorder Classifier with 10-fold cross_validation"
    main()
