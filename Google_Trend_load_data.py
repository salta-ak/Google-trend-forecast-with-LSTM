
import os
import pandas as pd
import numpy as np


# Load data from with pytrend API

from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)


kw_list = ["Bitcoin","flu","sale", 'rent']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')
data=pytrends.interest_over_time()
series = data.iloc[: , :-1].astype(float).sort_index()
train, test = train_test_split(series, test_size=0.1)
    
#data.to_csv('google_w.csv')

# Split data 

def train_test_split(data, test_size=0.1):
    split_row = len(data) - int(test_size * len(data))
    train_data = data.iloc[:split_row]
    test_data = data.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(series, test_size=0.1)

#train, test = train_test_split(series, test_size=0.1)
#train.to_csv('train.csv')
#train.to_csv('train.csv')
