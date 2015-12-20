#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################
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
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def getDataInFormat():
	dataTuples=[]
	fileobject=codecs.open("200g.all", "r", "utf-8")
	lstOfLines = fileobject.readlines()
	for line in lstOfLines:
		tag = line.split("\t")[0]
		#labelCount[tag]=labelCount[tag]+1
		#if(labelCount[tag]<13):
		tweets = line.split("\t")[3]
			#tweets = tweets.split("/")
			#for tweet in tweets:
		dataTuples.append([tag,tweets])
			
	
	return dataTuples

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
            if(label.find('P') != -1):
                    labels.append("Perceiving")
            else:
                    labels.append("Judging")
            
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
    trainTuples=dataTuples[:1000]
    testTuples=dataTuples[1000:]
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

    del trainVectors
    del trainLabels

    ids, labels, vectors= getLabelsAndVectors(testTuples)
    oneHotVectors=getOneHotVectors(ids, labels, vectors, spaceWithMetaFeatures, followerCountsList)
    vectors, labels=getOneHotVectorsAndLabels(oneHotVectors)
    del oneHotVectors
    testVectors = vectors
    testLabels = labels
    del labels
    del vectors
    predicted_testLabels = clf.predict(testVectors)
    print "Done predicting on DEV data...\n"
    print "classification_report:\n", classification_report(testLabels, predicted_testLabels)
    print "accuracy_score:", round(accuracy_score(testLabels, predicted_testLabels), 2)
    print "\n confusion_matrix:\n", confusion_matrix(testLabels, predicted_testLabels)
	
	
if __name__ == "__main__":
    print "OneVsRest Classifier with metafeature followerCount"
    main()
