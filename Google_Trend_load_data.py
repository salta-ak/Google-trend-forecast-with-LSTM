
import os
import pandas as pd
import numpy as np


# Load data from with pytrend API

kw_all=[kw_list,btc_list,st_list,cov_list]
kw_list = ["bitcoin","COVID-19","stock price"]
btc_list=["bitcoin","cryptocurrency","blockchain+bitcoin"]
st_list=["stock+price","stock+index","futures+stock"]
cov_list=["Covid","cough+covid","symptoms+covid"]
for i in kw_all:
    pytrends.build_payload(i, cat=0, timeframe='today 5-y', geo='', gprop='')
    data=pytrends.interest_over_time()
    data=data.loc[~(data==0.0).all(axis=1)]
    series = data.iloc[: , :-1].astype(float).sort_index()
    series.to_csv('{}.csv'.format(i))
    
    
