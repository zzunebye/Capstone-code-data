import pandas as pd
import nltk
import numpy as np
import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# from nltk.stem.wordnet import WordNetLemmatizer
from nltk import SnowballStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
""" Replaces contractions from a string to their equivalents """
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def getTokenization(raw_data):

    lmt = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    freqdist = nltk.FreqDist()
    tweet_tokenizer = TweetTokenizer()
    tweet_tokens = []
    stop_words = set(stopwords.words('english'))

    for sent in raw_data.text:

        sent = re.sub(r"http\S+", "&", sent)
        # sent = re.sub(r"@\S+", "@", sent)
        sent = re.sub(r"(#)(\S+)", r'\1 \2', sent)

        sent = re.sub(r'([^\s\w@#&]|_)+', '', sent)
        sent = re.sub('@[^\s]+','atUser',sent)
        # sent = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',sent)
        # sent = re.sub(r'#([^\s]+)', r'\1', sent)

        sent = replaceContraction(sent)

        # sent = re.sub('', '', sent.lower())
        # sent = [tweet_tokenizer.tokenize(sent)]
        sent = tweet_tokenizer.tokenize(sent.lower())
        sent = [stemmer.stem(token) for token in sent]
        # sent = [lmt.lemmatize(token) for token in sent]

        temp = [token for token in sent if not token in stop_words]
        tweet_tokens.append([temp])
        # tweet_tokens.append(tweet_tokenizer.tokenize(sent))
    df_tokens = pd.DataFrame(tweet_tokens, columns=['token'])


def get_W2V_AVG(raw_data):
    tweet_tokenizer = TweetTokenizer()
    tweet_tokens = []
    stop_words = set(stopwords.words('english'))

    for sent in raw_data.text:
        sent = re.sub(r"http\S+", "&", sent)
        sent = re.sub(r"@\S+", "@", sent)
        sent = re.sub(r"#\S+", "#", sent)
        sent = re.sub(r'([^\s\w@#&]|_)+', '', sent)
        # sent = re.sub('', '', sent.lower())
        # print(tweet_tokenizer.tokenize(sent))
        # sent = [tweet_tokenizer.tokenize(sent)]
        sent = [tweet_tokenizer.tokenize(sent.lower())]
        temp = [token for token in sent[0] if not token in stop_words]
        tweet_tokens.append([temp])
        # tweet_tokens.append(tweet_tokenizer.tokenize(sent))
    df_tokens = pd.DataFrame(tweet_tokens, columns=['token'])
    df_tokens['token_vec'] = copy.deepcopy(df_tokens['token'])

    for index, sent in enumerate(df_tokens['token_vec']):
        df_tokens['token_vec'][index] = vectorize(sent).mean(axis=0)

    df_temp = pd.DataFrame(
        df_tokens['token_vec'].values.tolist()).add_prefix('vec_avg')

    df_tokens = df_tokens.join(df_temp).drop('token_vec', axis=1)
    return pd.DataFrame(df_tokens)


def vectorize(line):
    words = []
    for word in line:  # line - iterable, for example list of tokens
        try:
            w2v_idx = w2v_indices[word]
        except KeyError:  # if you does not have a vector for this word in your w2v model, continue
            words.append(list(np.zeros(200,)))
            continue
        words.append(list(w2v_vectors[w2v_idx]))
        if not word:
            words.append(None)

        if len(line) > len(words):
            continue
    return np.asarray(words)


def cv_events(data):
    NUM_EVENT = data.Event.unique().shape[0]
    EVENTS = data.Event.unique()

    cv_pd_list = []
    for i, d in enumerate(EVENTS):
        df1, df2 = [x for _, x in data.groupby(data['Event'] != d)]
        df1.reset_index(inplace=True, drop=True)
        df2.reset_index(inplace=True, drop=True)
        cv_pd_list.append([df1, df2])
    return cv_pd_list
