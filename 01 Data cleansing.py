"""
Created on Tue Aug 20 17:52:48 2019
@author: Jonas Gartenmeier
"""

import datetime as dt
from datetime import timedelta, date
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import statsmodels.api as sm
import urllib
import io

#%% Import of Manager table

#Skipping row 1 and eliminating certain columns
Manager_table_adj = pd.read_csv('Manager_Table.csv', low_memory=False, skiprows=1)
Manager_table_adj.drop(['MS_Description', 
                 'DOB', 
                 'DOD', 
                 'B1_year', 
                 'B2_year', 
                 'B3_year', 
                 'MBA1_year', 
                 'MBA2_year', 
                 'M2_year', 
                 'M3_year', 
                 'P1_year', 
                 'P2_year', 
                 'JD_year', 
                 'MD_year',
                 'Father_Name',
                 'Father_DOB',
                 'Father_DOD',
                 'Mother_Name',
                 'Mother_DOB',
                 'Mother_DOD',
                 'Wedding',
                 'Spouses_Names',
                 'Obituary', 
                 'Relationship_Hint', 
                 'Candidates', 
                 ' BB_Description', 
                 'BB_WorksforCapID', 
                 'BB_AffiliationIDs'
                 ], axis=1, inplace = True)

'Eliminate/replace wordings in order to split relationship information'
Manager_table_adj['Relationship_ID'].replace(regex = True, inplace = True, to_replace = 'of ', value = '')
Manager_table_adj['Relationship_ID'].replace(regex = True, inplace = True, to_replace = 'to ', value = '')

'Split of relationship strings into multiple columns as multiple relationships exist'
Relation_split = Manager_table_adj['Relationship_ID'].str.split(' ', n=1, expand = True)
Manager_table_adj['Relation_class_1'] = Relation_split[0]
Manager_table_adj['Relation_ID_1'] = Relation_split[1]

'Replace misspellings'
Manager_table_adj['Relation_class_1'].replace(regex = True, inplace = True, to_replace = 'Realted', value = 'Related')
Manager_table_adj['Relation_class_1'].replace(regex = True, inplace = True, to_replace = 'Daugther', value = 'Daughter')

'Further splitting relationship data'
Relation_split_2 = Manager_table_adj['Relation_ID_1'].str.split('/', expand = True)

'Attaching newly split columns to existing dataframe'
Manager_table_adj['Relation_ID_1'] = Relation_split_2[0]
Manager_table_adj['Relation_ID_2'] = Relation_split_2[1]
Manager_table_adj['Relation_ID_3'] = Relation_split_2[2]
Manager_table_adj['Relation_ID_4'] = Relation_split_2[3]

'Further splitting relationship data'
Relation_split_3 = Manager_table_adj['Relation_ID_2'].str.split(' ', expand = True)
Manager_table_adj['Relation_class_2'] = Relation_split_3[0]
Manager_table_adj['Relation_ID_5'] = Relation_split_3[1]

Relation_split_4 = Manager_table_adj['Relation_ID_3'].str.split(' ', expand = True)
Manager_table_adj['Relation_class_3'] = Relation_split_4[0]
Manager_table_adj['Relation_ID_6'] = Relation_split_4[1]

'Removing duplicates that exist due to splitting of columns. Removing entries with characters only.'
Manager_table_adj['Relation_ID_2'] = pd.to_numeric(Manager_table_adj['Relation_ID_2'].astype(str).str.replace('',''), errors = 'coerce')
Manager_table_adj['Relation_ID_3'] = pd.to_numeric(Manager_table_adj['Relation_ID_3'].astype(str).str.replace('',''), errors = 'coerce')

'Removing duplicates that exist due to splitting of columns. Removing entries with numericals only.'
Manager_table_adj['Relation_class_2'] = Manager_table_adj['Relation_class_2'].mask(pd.to_numeric(Manager_table_adj['Relation_class_2'], errors = 'coerce').notna())
Manager_table_adj['Relation_class_3'] = Manager_table_adj['Relation_class_3'].mask(pd.to_numeric(Manager_table_adj['Relation_class_3'], errors = 'coerce').notna())

'Removal of column Relationship_ID'
Manager_table_adj.drop(columns = ['Relationship_ID'], inplace = True)
    
#Manager_table_adj.to_csv('Manager_Table_adjusted.csv')

#%% Import of Solo fund panel to match fund flows, returns, TNA, etc.
Solo_fund_table = pd.read_csv('Solo_fund_panel.csv')  

Solo_fund_table['date_crsp'] = pd.to_datetime(Solo_fund_table['date_crsp'])

#Necessary step so year 1968 does not get recognized as 2068
future = Solo_fund_table['date_crsp'] > date(year=2050, month=1, day=1)
Solo_fund_table.loc[future, 'date_crsp'] -= timedelta(days = 365.25*100)

#Alteration of columns
Solo_fund_table[['date_crsp', 'first_offer_dt_crsp']] = Solo_fund_table[['date_crsp', 'first_offer_dt_crsp']].apply(pd.to_datetime, errors = 'coerce')

Solo_fund_table = Solo_fund_table.rename(columns = {'ManagerID' : 'MS_ManagerID'})

Solo_fund_table.drop(['actual_12b1_crsp', 'successful_crsp_ms_merge', 'Roots', 'Generation'], axis=1, inplace = True)

# Removal of index funds & other columns
index_funds = Solo_fund_table[Solo_fund_table['index_fund_flag_crsp'] == 1].index
Solo_fund_table.drop(index_funds, inplace = True)

# Removal of funds that show no Manager ID
Solo_fund_table = Solo_fund_table.dropna(subset = ['MS_ManagerID'])

#print(Solo_fund_table.head())
print(Solo_fund_table.dtypes)
#print(Solo_fund_table.tail())

#%% Merger of Solo fund panel and Manager table
Family_table = pd.merge(Manager_table_adj, Solo_fund_table, on = 'MS_ManagerID')

#%% Download of Fama-French Dataset for Risk Factors (MKT-RF, SMB, HML)
          #https://github.com/nakulnayyar/FF3Factor/blob/master/FamaFrench3Factor.ipynb
          
url = urllib.request.urlopen("http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")

#Download Zipfile and create pandas DataFrame
zipfile = ZipFile(io.BytesIO(url.read()))
FF_3F = pd.read_csv(zipfile.open('F-F_Research_Data_Factors.CSV'), 
                     header = 0, names = ['Date','MKT-RF','SMB','HML','RF'], 
                     skiprows=9)

#Drop last row of data - String
FF_3F = FF_3F[:1104]
FF_3F['Date'] = pd.to_datetime(FF_3F['Date'], format = '%Y%m')

FF_3F['Date'] = FF_3F['Date'].map(lambda x: x.strftime('%d-%m-%Y'))

FF_3F['Date'] = pd.to_datetime(FF_3F['Date'], format = '%d-%m-%Y')

FF_3F = FF_3F.rename(columns = {'Date' : 'date_crsp_x'})

print(FF_3F.head())
print(FF_3F.dtypes)
print(FF_3F.tail())

#%% Download of Fama-French Dataset for Risk Factors (MOM)
          #https://github.com/nakulnayyar/FF3Factor/blob/master/FamaFrench3Factor.ipynb
          
url = urllib.request.urlopen('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip')

#Download Zipfile and create pandas DataFrame
zipfile = ZipFile(io.BytesIO(url.read()))
FF_MOM = pd.read_csv(zipfile.open('F-F_Momentum_Factor.CSV'), 
                     header = 0, names = ['Date','MOM'], 
                     skiprows=13)

#Drop last row of data - String
FF_MOM = FF_MOM[:1104]
FF_MOM['Date'] = pd.to_datetime(FF_MOM['Date'], format = '%Y%m')

FF_MOM['Date'] = FF_MOM['Date'].map(lambda x: x.strftime('%d-%m-%Y'))

FF_MOM['Date'] = pd.to_datetime(FF_MOM['Date'], format = '%d-%m-%Y')

FF_MOM = FF_MOM.rename(columns = {'Date' : 'date_crsp_x'})

print(FF_MOM.head())
print(FF_MOM.dtypes)
print(FF_MOM.tail())

#%% Joining the two CSV files

FF_all = FF_3F.join(FF_MOM.set_index('date_crsp_x'), on = 'date_crsp_x')

FF_all = FF_all[['date_crsp_x', 'MKT-RF', 'SMB', 'HML', 'MOM', 'RF']]

#Convert into float
FF_all['MKT-RF'] = FF_all['MKT-RF'].astype('float')
FF_all['SMB'] = FF_all['SMB'].astype('float')
FF_all['HML'] = FF_all['HML'].astype('float')
FF_all['MOM'] = FF_all['MOM'].astype('float')
FF_all['RF'] = FF_all['RF'].astype('float')

print(FF_all.head())
print(FF_all.dtypes)
print(FF_all.tail())

#FF_all.to_csv('FFF.csv')

#%% If necessary, import and merger of S&P 500 data

#SP500 = pd.read_csv('S&P500.csv')

#SP500 = SP500.rename(columns = {'caldt' : 'date_crsp'})
#SP500[['date_crsp']] = SP500[['date_crsp']].apply(pd.to_datetime, errors = 'coerce')

# Merged table on month, not exact date due to differences in recordings. Code from https://stackoverflow.com/questions/51992291/how-to-join-2-dataframe-on-year-and-month-in-pandas
#Family_table = pd.merge(Family_table.assign(date_crsp_m = Family_table['date_crsp'].dt.to_period('M')), SP500.assign(date_crsp_m = SP500['date_crsp'].dt.to_period('M')), on = 'date_crsp_m', how = 'left')

#Family_table = Family_table.drop(columns = ['date_crsp_m', 'date_crsp_y'])

#print(Family_table.head())
#print(Family_table.dtypes)
#print(Family_table.tail())

#%% Last merger of dataframes 

# Merged table on month, not exact date due to differences in recordings. Code from https://stackoverflow.com/questions/51992291/how-to-join-2-dataframe-on-year-and-month-in-pandas
Family_table = pd.merge(Family_table.assign(dates = Family_table['date_crsp'].dt.to_period('M')), FF_all.assign(dates = FF_all['date_crsp_x'].dt.to_period('M')), on = 'dates', how = 'left')

print(Family_table.head())
print(Family_table.dtypes)
print(Family_table.tail())

#Family_table.to_csv('Family_table.csv')