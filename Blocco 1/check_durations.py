import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime


# Creating dataframe
folder = 'C:/Users/giord/Downloads/only AC data/only AC/'
#folder = 'C:/Users/david/Documents/University/Tesi/Python AInCP/only AC/'
metadata = pd.read_excel(folder + 'metadata2022_04.xlsx')

datetime_df = pd.DataFrame(columns=['start csv','end csv','start meta','end meta'])

#
#   start AHA   |   end AHA | start metadata    | end metadata
#-------------------------------------------------------------------
#



for i in range (1,61):

    df = pd.read_csv(folder + 'data/' + str(i) + '_AHA_1sec.csv')

    start_csv = (datetime.strptime(df['datetime'].iloc[0], '%Y-%m-%d %H:%M:%S'))
    end_csv = (datetime.strptime(df['datetime'].iloc[-1], '%Y-%m-%d %H:%M:%S'))

    #start_meta = (datetime.strptime(metadata['start AHA'].iloc[i], '%H:%M:%S'))
    #end_meta = (datetime.strptime(metadata['end AHA'].iloc[i], '%H:%M:%S'))
    start_meta = metadata['start AHA'].iloc[i-1]
    end_meta = metadata['stop AHA'].iloc[i-1]

    print('paziente: ',i,' CSV: ' ,start_csv.time() ,' - ',end_csv.time(), ' META: ',str(start_meta),' - '+str(end_meta))
