import pandas as pd
import numpy as np
from plots import *

def Morris(df):
    df2=df.copy()
    df2['Morris'] = (1200/np.log10(df2['T2lm (ms)']))**0.9
    df2.dropna(subset='Morris', axis=0, inplace=True)
    # print(df2['Morris'].isna().value_counts(), df2.shape)
    regression_metrics_models2('Morris 1994', df2['Viscosity (cP)'], df2['Morris'])

def Vinegar(df):
    df2=df.copy()
    df2['Vinegar'] = 4.03*(df2['Temperature (K)']/(df2['T2lm (ms)']))
    df2.dropna(subset='Vinegar', axis=0, inplace=True)
    # print(df2['Vinegar'].isna().value_counts(), df2.shape)
    regression_metrics_models2('Vinegar 1996', df2['Viscosity (cP)'], df2['Vinegar'])

def Lo(df):
    df2=df.copy()
    df2['Lo'] = 9.558*(df2['Temperature (K)']/(df2['T2lm (ms)']))
    df2.dropna(subset='Lo', axis=0, inplace=True)
    # print(df2['Lo'].isna().value_counts(), df2.shape)
    regression_metrics_models2('Lo 2001', (df2['Viscosity (cP)']), df2['Lo'])

def Zhang(df):    
    df2=df.copy()
    df2['Zhang'] = 7.13*(df2['Temperature (K)']/(df2['T2lm (ms)']))
    df2.dropna(subset='Zhang', axis=0, inplace=True)
    # print(df2['Zhang'].isna().value_counts(), df2.shape)
    regression_metrics_models2('Zhang 2001', df2['Viscosity (cP)'], df2['Zhang'])

def Sandor(df):
    df2=df.copy()
    df2['Sandor'] = df2['Temperature (K)']*(9.2/((df2['T2lm (ms)'])-(0.69*df2['TE (ms)'])-0.86))**(1/((-0.087*df2['TE (ms)'])+0.90))
    df2.dropna(subset='Sandor', axis=0, inplace=True)
    # print(df2['Sandor'].isna().value_counts(), df2.shape)
    regression_metrics_models2('Sandor 2016', df2['Viscosity (cP)'], df2['Sandor'])

def Straley(df):
    df2=df.copy()
    df2['Straley']=(10**(1.2/(df2['T2lm (ms)'])))**(1/9)
    # print(df2['Straley'].isna().value_counts(), df2.shape)
    df2.dropna(subset='Straley', axis=0, inplace=True)
    regression_metrics_models2('Straley 1997', df2['Viscosity (cP)'], df2['Straley'])

def Cheng(df):
    df2=df.copy()
    df2['Cheng']= df2['Temperature (K)']*(((df2['T2lm (ms)'])-0.3682)/5.8235)**(1/-0.6139)
    # print(df2['Cheng'].isna().value_counts(), df2.shape)
    df2.dropna(subset='Cheng', axis=0, inplace=True)
    regression_metrics_models2('Cheng 2009', df2['Viscosity (cP)'], df2['Cheng'])