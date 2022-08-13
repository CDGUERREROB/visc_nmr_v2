import numpy as np
import pandas as pd

def process_data(): 
    train_data = pd.read_csv("../data/raw/train_data.csv", index_col=0)
    test_markovic = pd.read_csv("../data/raw/test_markovic.csv", index_col=0)
    test_data = pd.read_csv("../data/raw/test_data.csv", index_col=0)

    data = pd.concat(
        [
        train_data,
        test_markovic,
        test_data
        ]
        ).reset_index(drop=True)

    data['T (K)'] =  data['T (K)'].apply(lambda x : (x - 273.15)*(9/5)+ 32 )

    def rename(df):
        df.rename(
        columns = {
        'T2LM (ms)' : 'T2lm (ms)',
        'T (K)':'Temperature (°F)',
        'Eta (cP)' : 'Viscosity (cP)'
        },
        inplace = True
        )
        return df

    for i in [train_data,  test_data, data]:
        i['Temperature (K)'] = i['T (K)']
        rename(i)

    # data.drop(index=[205, 206, 207, 208, 209, 210, 294, 295], axis=0, inplace=True)

    return train_data, test_data, data

def well_log_data():
    well_log = pd.read_excel("../data/processed/CASN0193.xlsx")
    return well_log