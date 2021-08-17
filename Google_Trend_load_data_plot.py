
import os
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360)

# Load data from with pytrend API


filename=['bitcoin','covid','stock_price']
kw_list = ["bitcoin","COVID-19","stock price"]
btc_list=["bitcoin","cryptocurrency","blockchain+bitcoin"]
st_list=["stock+price","stock+index","futures+stock"]
cov_list=["Covid","cough+covid","symptoms+covid"]
kw_all=[kw_list,btc_list,st_list,cov_list]
count=0
figure, axes = plt.subplots(4, 1, figsize=(15,15))
for i,j in zip(kw_all,filename):
    pytrends.build_payload(i, cat=0, timeframe='today 5-y', geo='', gprop='')
    data=pytrends.interest_over_time()
    data=data.loc[~(data==0.0).all(axis=1)]
    series = data.iloc[: , :-1].astype(float).sort_index()
    series.to_csv('{}.csv'.format(j))
    figure, axes = plt.subplots(1, 1)
    series.plot(ax=axes[count])
    count+=1
    
    
