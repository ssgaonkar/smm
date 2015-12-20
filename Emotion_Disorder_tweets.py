#Import the necessary methods from tweepy library
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "2796294980-5r5a9q88F9gJV304jcBCieEUnVk4CcO1An9Cbjs"
access_token_secret = "WaTZIesNluIHX8yleojUq0gnKW6T6pi9WJ4sNmNZzYCnV"
consumer_key = "YGq335O4dXisApcxsZbHr8erj"
consumer_secret = "cNABMPt9mad5Inqm0Q7ZuiePbQKYd30M3QNjjATud8KMDIyilO"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        try:
            #print data
            saveFile = open('emotion.txt','a')
            saveFile.write(data)
            saveFile.write('\n')
            saveFile.close()
            return True
        except BaseException, e:
            print 'failed ', str(e)

    def on_error(self, status):
        print status


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords	
    stream.filter(track=['#anxiety', '#anxious', '#suicidal', '#depressed', '#depression'])
