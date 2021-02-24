import pandas as pd
import numpy as np
from glob2 import glob
import json

import nltk
import re
import gensim
import gensim.models.word2vec as w2v
from datetime import datetime
from datetime import date
from datetime import timedelta

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def capitalratio(tweet_text):
        uppers = [l for l in tweet_text if l.isupper()]
        capitalratio = len(uppers) / len(tweet_text)
        return capitalratio 

def flatten_tweets(tweets):

    def tweets2tokens(tweet_text):

        # Tokenizing
        tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tweet_text.lower()))

        # Setting url value (whether the tweet contains http link) and filter http links
        url=0
        for token in tokens:
            if token.startswith( 'http' ):
                url=1
        tokens = [token for token in tokens if not token.startswith('http')]

        ## Stemming
        # porter = PorterStemmer()
        # tokens = [porter.stem(token) for token in tokens]

        # Filtering Stop words
        # from nltk.corpus import stopwords
        # stop_words = set(stopwords.words('english'))
        # tokens = [token for token in tokens if not token in stop_words]

        return tokens,url
    
    def getposcount(tokens):
            postag = []
            poscount = {}
            poscount['Noun']=0
            poscount['Verb']=0
            poscount['Adjective'] = 0
            poscount['Pronoun']=0
            poscount['FirstPersonPronoun']=0
            poscount['SecondPersonPronoun']=0
            poscount['ThirdPersonPronoun']=0
            poscount['Adverb']=0
            poscount['Numeral']=0
            poscount['Conjunction_inj']=0
            poscount['Particle']=0
            poscount['Determiner']=0
            poscount['Modal']=0
            poscount['Whs']=0
            Nouns = {'NN','NNS','NNP','NNPS'}
            Adverbs = {'RB','RBR','RBS'}
            Whs = {'WDT','WP','WRB'} # Composition of wh-determiner(that,what), wh-pronoun(who), wh-adverb(how)
            Verbs={'VB','VBP','VBZ','VBN','VBG','VBD','To'}
            first_person_pronouns=['i','I','me','my','mine','we','us','our','ours'] #'i',
            second_person_pronouns=['you','your','yours']
            third_person_pronouns=['he','she','it','him','her','it','his','hers','its','they','them','their','theirs']

            for word in tokens:
                w_lower=word.lower()
                if w_lower in first_person_pronouns:
                    poscount['FirstPersonPronoun']+=1
                elif w_lower in second_person_pronouns:
                    poscount['SecondPersonPronoun']+=1
                elif w_lower in third_person_pronouns:
                    poscount['ThirdPersonPronoun']+=1
            
            postag = nltk.pos_tag(tokens)
            for g1 in postag:
                if g1[1] in Nouns:
                    poscount['Noun'] += 1
                elif g1[1] in Verbs:
                    poscount['Verb']+= 1
                elif g1[1]=='ADJ'or g1[1]=='JJ':
                    poscount['Adjective']+=1
                elif g1[1]=='PRP' or g1[1]=='PRON' or g1[1]=='PRP$':
                    poscount['Pronoun']+=1
                elif g1[1] in Adverbs or g1[1]=='ADV':
                    poscount['Adverb']+=1
                elif g1[1]=='CD':
                    poscount['Numeral']+=1
                elif g1[1]=='CC' or g1[1]=='IN':
                    poscount['Conjunction_inj']+=1
                elif g1[1]=='RP':
                    poscount['Particle']+=1
                elif g1[1]=='MD':
                    poscount['Modal']+=1
                elif g1[1]=='DT':
                    poscount['Determiner']+=1
                elif g1[1] in Whs:
                    poscount['Whs']+=1
            return poscount

    def contentlength(words):
        wordcount = len(words)
        return wordcount

    """ Flattens out tweet dictionaries so relevant JSON is in a top-level dictionary. """
    tweets_list = []
    total_tokens_l = []

    # Iterate through each tweet
    for tweet_obj in tweets:
        output_f = dict()

        output_f['text']= tweet_obj['text']
        output_f['text_token'], _ = tweets2tokens(tweet_obj['text'])
        total_tokens_l.extend(output_f['text_token']) # append the tokens to list of total tokens

        '''POS Tagging'''
        pos_dict=getposcount(output_f['text_token'])
        output_f.update(pos_dict)

        output_f['char_count'] = len(output_f['text'])
        output_f['word_count'] = len(output_f['text_token'])
        output_f['has_question'] = "?" in output_f["text"]
        output_f['has_exclaim'] = "!" in output_f["text"]
        output_f['has_period'] = "." in output_f["text"]
    
        ''' User info'''
        # Store the user screen name in 'user-screen_name'
        # output_f['user-screen_name'] = tweet_obj['user']['screen_name']
        
        # Store the user location
        # output_f['user-location'] = tweet_obj['user']['location']

        acc_created = datetime.strptime(tweet_obj['user']['created_at'], '%a %b %d %H:%M:%S %z %Y')
        tweet_created = datetime.strptime(tweet_obj['created_at'], '%a %b %d %H:%M:%S %z %Y')
        age = (tweet_created - acc_created)
        # print(type(timedelta.total_seconds(age)))

        output_f['capital_ratio']=(capitalratio(tweet_obj['text']))
        output_f['tweet_count'] = np.log10(tweet_obj['user']['statuses_count'])
        output_f['listed_count'] = np.log10(tweet_obj['user']['listed_count'])
        output_f['follow_ratio'] = np.log10(tweet_obj['user']['followers_count'])
        output_f['age'] = int(timedelta.total_seconds(age)/86400)
        output_f['verified'] = tweet_obj['user']['verified']

        tweets_list.append(output_f)

    unk_tokens_l = list(set(total_tokens_l))
    print(len(total_tokens_l), len(unk_tokens_l)) # number of tokens and unique tokens

    return tweets_list