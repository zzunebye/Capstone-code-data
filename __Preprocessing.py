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
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def fetchRawText(path, events, tweetType):
    jsons = []
    for i, event in enumerate(events):
        jsons.append(glob('%s/%s/**/%s/*.json' % (path, event,tweetType)))
    for i,d in enumerate(jsons): print("%s's length is %d" %(events[i], len(d)))

    targets = []
    features = []
    for index, dataset in enumerate(jsons):
        targetEvent = []
        dataEvent = []
        count = 0  # help var
        for jsonFile in dataset:
            count += 1
            if jsonFile.find("non-rumours") == -1:
                targetEvent.append(1)
            else:
                targetEvent.append(0)

            with open(jsonFile, 'r') as f:
                for l in f.readlines():
                    if not l.strip():  # skip empty lines
                        continue
                    json_data = json.loads(l)
                    # print (json_data,"\n\n")
                    dataEvent.append(json_data)
        print(index, events[index], len(targetEvent), len(dataEvent))
        targets.append(targetEvent)
        features.append(dataEvent)

    # print("\nNumber of Events:", len(targets))
    # print("Number of tweets in the first event:", len(targets[0]))

    # targets은 targetEvent들을 리스트에 담은 것
    target_list = []
    for event in targets:
        for elem in event:
            target_list.append(elem)
    target = pd.DataFrame(target_list, columns=["target"])

    extracted_features = []

    extracted = []

    for obj_list in features:
        extracted_event = []
        for obj in obj_list:
            output_f = dict()
            output_f['text'] = obj['text']
            extracted_event.append(output_f)
        extracted_features.append(extracted_event)

    extracted_df = []
    for i, data in enumerate(extracted_features):
        temp = pd.DataFrame(data)
        temp["Event"] = events[i]
        extracted_df.append(pd.DataFrame(temp))

    final = pd.concat(extracted_df, ignore_index=True)
    final = pd.concat([final, target], axis=1)
    return final



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
    second_person_pronouns=['you','your','yours', 'ya']
    third_person_pronouns=['he','she','it','him','her','it','his','hers','its','they','them','their','theirs']
    test_auxiliary=['be','will','have','am','is','was','were','can','could','dare','did','may','might','must','ought','shall','should','would']
    test_tentat=['maybe','perhaps','possibly','probably','guess']
    test_certain=['always','never', "can't", 'cannot']

    for word in tokens:
        w_lower=word.lower()
        if w_lower in first_person_pronouns:
            poscount['FirstPersonPronoun']+=1
        elif w_lower in second_person_pronouns:
            poscount['SecondPersonPronoun']+=1
        elif w_lower in third_person_pronouns:
            poscount['ThirdPersonPronoun']+=1
        
    for word in tokens:
        w_lower=word.lower()
        if w_lower in test_auxiliary:
            poscount['test_auxiliary']+=1
        elif w_lower in test_tentat:
            poscount['test_tentat']+=1
        elif w_lower in test_certain:
            poscount['test_certain']+=1

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



def text_preprocessing_simple(text, lemma=False, twttknzr=False): # Create a function to tokenize a set of texts
    from sklearn.ensemble import ExtraTreesClassifier

    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """

    # text = re.sub(r"http\S+", "*", text)  # http link -> '*'
    # sent = re.sub(r'([^\s\w@#\*]|_)+', '', sent) # Erasing Special Characters

    # text = emoji.demojize(text)
    # text = re.sub(r':[^:\s]*:', r' \g<0>', text)  # http link -> '*'

    # text = re.sub(r':[^:\s]*(?:::[^:\s]*)*:', r' \g<0> ', text)  # http link -> '*'

    # text = re.sub(r"\n", " ", text)   # mention -> '@'
    text = re.sub(r"http\S+", "HTTPURL", text)  # http link -> '*'
    # text = re.sub(r"@\S+", "@USER", text)   # mention -> '@'
    
    text = re.sub(r"@[A-Za-z0-9]+", "@USER", text)   # mention -> '@'

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'&amp;', '&', text)
    text = tweetTokenizer.tokenize(text)
    text = [emoji.demojize(token) for token in text]

    return text


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

def test_data_process(X_test, y_test):
    tensor_x1 = torch.Tensor(X_test.values).unsqueeze(1)
    tensor_y1 = torch.Tensor(y_test.values).unsqueeze(1)
    test_dataset = TensorDataset(tensor_x1,tensor_y1)

    batch_size = 16

    # train_sampler, test_sampler = __MLP.getSamplers(pheme_y, tensor_x2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    data = next(iter(test_dataloader))
    print("mean: %s, std: %s" %(data[0].mean(), data[0].std()))

    test_size = int(tensor_y1.size(0))

    print("Test Size",test_size)

    # predict_batch
    return test_dataloader, test_size

def f_imp(X, y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=3)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d: %s (%f)" % (f + 1, indices[f], X.columns[indices[f]], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    
    plt.figure(figsize=(12, 7))
    size=20
    params = {'legend.fontsize': 'large',
            'figure.figsize': (20,8),
            'axes.labelsize': size,
            'axes.titlesize': size,
            'xtick.labelsize': size*0.75,
            'ytick.labelsize': size*0.75,
            'axes.titlepad': 25}
    plt.rcParams.update(params)
    plt.title("Feature importances")
    plt.bar( X.columns[indices], importances[indices], color="r")
    plt.tight_layout()
    plt.show()

def f_imp2(X, y):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    result = permutation_importance(rf, X, y, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 7)
    ax.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=X.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)",fontdict={'fontsize': 'x-large', 'fontweight': 'medium'})
    ax.labelsize='x-large'
    fig.titlesize='x-large'
    fig.tight_layout()
    plt.show()