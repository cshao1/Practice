# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:47:36 2019

@author: cshao
"""

import numpy as np
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt

'def cross_above():
    
    
    
px = pd.read_csv("E:\OnlineStorage\Data\dataAll_2014-2015.zip")

    
sma = ta.SMA(px.close)
bband_upper, bband_middle, bband_lower = ta.BBANDS(px.close, matype=ta.MA_Type.T3)
mom = ta.MOM(px.close, timeperiod=5)

plt.figure(dpi=100)
pltRng = list(range(100,1000))
plt.plot(px.close[pltRng])
plt.plot(sma[pltRng])
#plt.plot(mom)
plt.plot(bband_upper[pltRng])
plt.plot(bband_middle[pltRng])
plt.plot(bband_lower[pltRng])