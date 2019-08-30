"""
Created on Tue Aug 20 17:52:48 2019
To stop a python script just press Ctrl + C 
To delete variable explorer press %reset
!pip install
@author: Jonas Gartenmeier
"""

import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import pandas as pd
import numpy as np
import scikits.statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.rolling import RollingOLS

#%% Import of previously cleansed data

# Family_table = pd.read_csv('Family_table.csv', low_memory = False)

# Removal of fund managers that have no relative working in fund industry, too 
#Family_table_2 = Family_table.dropna(subset = ['Relation_class_1'])

#%% Creating dummy variables for multivariate regression

Family_table_2 = Family_table

Family_table_2.fillna({'B1': 0, 'B2': 0, 'B3': 0, 'MBA1': 0, 'MBA2': 0, 'M1': 0, 
                     'M2': 0, 'M3': 0, 'P1': 0, 'P2': 0, 'JD': 0, 'MD': 0, 
                     'Relation_class_1': 'Unrelated', 'index_fund_flag_crsp': 0}, inplace = True)

Family_table_2 = pd.get_dummies(Family_table_2, columns = ['Gender', 'Relation_class_1'], drop_first = True)

print(Family_table_2.head())

#%% Calculating excess and (36-month trailing returns)

# 36-month trailing returns: not possible due to a limited number of observations
#Family_table['fund_mret_36m_avrg'] = Family_table.groupby(['MS_ManagerName', 'crsp_cl_grp_crsp']).fund_mret.rolling(window = 3).mean().sort_index(level = 1).values

#Excess returns
Family_table_2['XReturn'] = (Family_table_2['fund_mret']*100 - Family_table_2['RF'])
Family_table_2['Alpha'] = (Family_table_2['XReturn'] - Family_table_2['MKT-RF'] - Family_table_2['SMB'] - Family_table_2['HML'] - Family_table_2['MOM'] )

print(Family_table_2.head())
print(Family_table_2.dtypes)
print(Family_table_2.tail())

#%%

exog_vars = ['MKT-RF', 'SMB', 'HML', 'MOM']
endog = Family_table_2.XReturn
exog = sm.add_constant(Family_table_2[exog_vars])
rols = RollingOLS(endog, exog, window = 12)
rres = rols.fit()
params = rres.params
print(params.head())
print(params.tail())

#%%

def ols_coef(x,formula):
    return smf.ols(formula,data=x).fit().params

gamma = (Family_table_2.groupby('crsp_cl_grp_crsp').apply(ols_coef,'XReturn ~ 1 + SMB + HML + MOM'))
gamma.head()    

#%%
def ols_res(df, xcols,  ycol):
    return sm.OLS(df[ycol], df[xcols]).fit().predict()

Family_table_2['TTE'] = Family_table_2.groupby('crsp_cl_grp_crsp').apply(lambda x: x.rolling(window = 3, min_periods = 1)).apply(ols_res, xcols=['MKT-RF', 'SMB'], ycol='XReturn')

#%%

#Family_table_2.shape
#Family_table_2.describe()

#def ols_res(df, xcols, ycol):
#    return sm.OLS(df[ycol], df[xcols]).fit().predict()

#y = Family_table_2.groupby('crsp_cl_grp_crsp').apply(ols_res, xcols = ['MKT-RF', 'SMB', 'HML', 'MOM'], ycol = 'XReturn')

def ols_predict(indices, result, ycol, xcols):
    roll_df = df.loc[indices] # get relevant data frame subset
    result[indices[-1,:]] = sm.OLS(roll_df[ycol], roll_df[xcols]).fit().predict()
    return 0 # value is irrelvant here

# define kwargs to be fet to the ols_predict
kwargs = {'xcols': ['MKT-RF', 'SMB', 'HML', 'MOM'], 
          'ycol': 'XReturn', 'result': {}}

# iterate id's sub data frames and call ols for rolling windows
Family_table_2['crsp_cl_grp_crsp'] = Family_table_2.index
for idx, sub_df in Family_table_2.groupby('crsp_cl_grp_crsp'):
    sub_df['crsp_cl_grp_crsp'].rolling(12, min_periods=6).apply(ols_predict, kwargs = kwargs)

# write results back to original df
Family_table_2["parameters"] = pd.Series(kwargs["result"])

# showing the last 5 computed values
print(Family_table_2["parameters"].tail())

#%%
return_per_class = Family_table.groupby(['MS_ManagerName', 'Relation_class_1', 'crsp_cl_grp_crsp']
    ).agg(
           {'fund_mret': ['count', 'mean', min, max, 'median']} )
  
print(return_per_class)    

#return_class_son['mean'].plot(kind='line',
         #  color = 'b',
          #figsize=(16,10))

#%% Analysis of selected funds [Distribution by date]

    # Distribution by date(Family_table_m['fund_mret']

(Family_table['XReturn']
    .groupby([
         Family_table['date_crsp_x_x'].dt.year,
         #Family_table['date_crsp_x'].dt.month
     ])
     .count()
     .plot(kind='bar',
           color = 'b',
          figsize=(20,10)))

#%% Analysis of selected funds [Observations per manager]

    # Distribution by date(Family_table_m['fund_mret']
    
a = (Family_table['XReturn']
    .groupby([
         Family_table['MS_ManagerID']
     ])
    .count()
)
a = a.groupby(a).count().cumsum().T

a.plot(kind='line',
       color = 'b',
       figsize=(10,5))

#%% Aggregate returns

#return_class_son = Family_table_m[Family_table_m['Relation_class_1'] == 'Son']

class_returns = Family_table.groupby('Relation_class_1')['XReturn'].mean()
print(class_returns)

#overall_return = Family_table_m3('Relation_class_1')['fund_mret'].mean()

#%% OLS Regression

y = Family_table['XReturn']
X = Family_table.loc[:,['MKT-RF', 'SMB', 'HML', 'MOM']]
X = sm.add_constant(X)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

#%% Correlation heat map to show that metric/ordinal variables are uncorrelated

f, ax = plt.subplots(figsize=(10, 8))
corr = Family_table.corr().loc[['fund_mret', 'fund_flows', 'fund_mtna', 'MKT-RF', 'SMB', 'HML', 'MOM'],['fund_mret', 'fund_flows', 'fund_mtna', 'MKT-RF', 'SMB', 'HML', 'MOM']]
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(220, 10, as_cmap = True),
            square = True, ax = ax)

#%% Boxplot images of Relation class

fig, ax = plt.subplots(figsize = (20,10))

boxprops = {'edgecolor': 'k', 'linewidth': 1, 'facecolor': 'w'}
lineprops = {'color': 'k', 'linewidth': 1}

kwargs = {'palette': pal}

boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,
                       'whiskerprops': lineprops, 'capprops': lineprops,
                       #'width': 1
                       },
                      **kwargs)

stripplot_kwargs = dict({'linewidth': 0.5, 'size': 1, 'alpha': 0.4},
                        **kwargs)

sns.boxplot(x = 'Relation_class_1', y = 'XReturn', hue = 'Relation_class_1', ax = ax, 
            fliersize = 0, data = Family_table, **boxplot_kwargs)

sns.stripplot(x = 'Relation_class_1', y = 'XReturn', hue = 'Relation_class_1', ax = ax,
            data = Family_table, jitter = 0.1, dodge = False, **stripplot_kwargs)

ax.legend_.remove()

#plt.savefig('Boxplot_relation_classes.jpeg', dpi = 1000)

#boxplot = Family_table.boxplot(column = ['XReturn'])
