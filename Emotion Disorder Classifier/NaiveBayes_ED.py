import nltk
import json
import codecs
from nltk import *

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
	
def isRetweet(tweet):
    if tweet.lower().split()[0] =="rt":
        return True
    return False

docs = []
fileobject=codecs.open("tweets.txt", "r", "utf-8")
lstOfLines = fileobject.readlines()
for line in lstOfLines:
        json_obj = json.loads(line)
        for key,value in json_obj.iteritems():
                if(key =='text'):
                        tweet = value
                        if(isRetweet(tweet)==False):
                                label = getLabel(tweet)
                                tweet = processTweet(tweet)
                                docs.append([tweet,label])

print len(docs)
        
all_words = nltk.FreqDist(w.lower() for w in list(set(word_tokenize(str(docs)))))
wfeatures = list(all_words)


def document_feature(document):
    dwords = set(document)
    features = {}
    for word in wfeatures:
        features['contains(%s)' % word] = (word in dwords)
    return features

featuresets = []
for i in range(len(docs)):
        a = document_feature(docs[i][0].split(' '))      
        featuresets.append((a,docs[i][1]))


train_set, test_set = featuresets[:2000], featuresets[2000:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print "Accuracy: " , nltk.classify.accuracy(classifier, test_set)

classifier.show_most_informative_features(20)





