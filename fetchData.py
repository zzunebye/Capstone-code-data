import pandas as pd
import numpy as np
import random

# define the function blocks
'''
@Options
    target
    bert
    sparse
    target
    avgw2v
    token
'''
def fetchdata(dataset, info):
    options = {
        'pheme':{
            'text': "./data/_PHEME_text.csv",
            'target': "./data/_PHEME_target.csv",
            'token': "./data/_PHEME_text_twtToken.csv",
            'sparse': "./data/_PHEME_sparse.csv",
            'avgw2v': "./data/_PHEME_text_AVGw2v.csv",
            'bert': "./data/_PHEME_bert.csv",
            'thread': "./data/all/_PHEME_thread.csv"
        },
        'ext':{
            'text': "./data/_PHEMEext_text.csv",
            'target': "./data/_PHEMEext_target.csv",
            'token': "./data/_PHEMEext_text_twtToken.csv",
            'sparse': "./data/_PHEMEext_sparse.csv",
            'avgw2v': "./data/_PHEMEext_text_AVGw2v.csv",
            'bert': "./data/_PHEMEext_bert.csv",
            'thread': "./data/all/_PHEMEext_thread.csv"
        },
        'rhi':{
            'text': "./data/_RHI_text.csv",
            'target': "./data/_RHI_target.csv",
            'token': "./data/_RHI_text_twtToken.csv",
            'avgw2v': "./data/_RHI_text_AVGw2v.csv",
            'bert': "./data/_RHI_bert.csv"
        }
    }
    return pd.read_csv(options[dataset][info])

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

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
