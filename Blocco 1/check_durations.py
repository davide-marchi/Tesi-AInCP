import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime


# Creating dataframe
#folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')

datetime_df = 

#
#   start AHA   |   end AHA | start metadata    | end metadata
#-------------------------------------------------------------------
#



for i in range (1,61):

    df = pd.read_csv(folder + 'data/' + str(i) + '_AHA_1sec.csv')

    for str in df['datetime']:

        date.append(datetime.strptime(str, '%Y-%m-%d %H:%M:%S'))

    date.append(datetime.strptime(df['datetime'].iloc[0], '%Y-%m-%d %H:%M:%S'))
    date.append(datetime.strptime(df['datetime'].iloc[-1], '%Y-%m-%d %H:%M:%S'))