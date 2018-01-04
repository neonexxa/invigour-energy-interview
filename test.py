
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import math 

def rmse(actual, predict):
    return sqrt(mean_squared_error(actual, predict))

def rmsle(y, y_pred):
    return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y + 1)).mean())

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[12]:


ori_df = pd.read_csv("train.csv")
ori_df['pickup_datetime'] = pd.to_datetime(ori_df['pickup_datetime'])
ori_df['dropoff_datetime'] = pd.to_datetime(ori_df['dropoff_datetime'])


# In[33]:


def get_season(month):
    if month >= 3 and month <= 5:
        return 'spring'
    elif month >= 6 and month <= 8:
        return 'summer'
    elif month >= 9 and month <= 11:
        return 'autumn'
    else:
        return 'winter'

ori_df = ori_df.assign(pickup_time=pd.cut(ori_df.pickup_datetime.dt.hour, [-1, 12, 16, 24], labels=['Morning', 'Afternoon', 'Evening']))
ori_df = ori_df.assign(dropoff_time=pd.cut(ori_df.dropoff_datetime.dt.hour, [-1, 12, 16, 24], labels=['Morning', 'Afternoon', 'Evening']))

ori_df['pickup_Month'] = ori_df['pickup_datetime'].map(lambda x: x.month)
ori_df['dropoff_Month'] = ori_df['dropoff_datetime'].map(lambda x: x.month)
ori_df['pickup_season'] = ori_df['pickup_Month'].apply(get_season)
ori_df['dropoff_season'] = ori_df['dropoff_Month'].apply(get_season)


# In[34]:


ori_df.head()


# In[35]:


ori_df.info()


# In[36]:


ori_df.describe()


# Max Trip Duration ? **58,771.36 minutes** or **979.52 hours**
# 
# Mean Trip Duration ? **15.99 minutes** 

# In[37]:


ori_df.columns


# In[44]:


# Data Prep
x = ori_df.drop(['id', 'vendor_id', 'pickup_datetime', 
            'dropoff_datetime', 'pickup_longitude', 
            'dropoff_longitude', 'trip_duration',
            'dropoff_latitude', 'pickup_latitude'], axis=1)
y = ori_df["trip_duration"]


# In[50]:


# Encode Label / Dummy coding
le_flag = preprocessing.LabelEncoder()
x["store_and_fwd_flag"] = le_flag.fit_transform(x["store_and_fwd_flag"])

le_pseason = preprocessing.LabelEncoder()
x["pickup_season"] = le_pseason.fit_transform(x["pickup_season"])

le_dseason = preprocessing.LabelEncoder()
x["dropoff_season"] = le_dseason.fit_transform(x["dropoff_season"])

le_ptime = preprocessing.LabelEncoder()
x["pickup_time"] = le_ptime.fit_transform(x["pickup_time"])

le_dtime = preprocessing.LabelEncoder()
x["dropoff_time"] = le_dtime.fit_transform(x["dropoff_time"])


# In[51]:


x.head()


# In[52]:


y.head()


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[55]:


dtr = DecisionTreeRegressor()
dtr = dtr.fit(X_train,y_train)
y_pred = dtr.predict(X_test)


# In[89]:

print("Decision Tree")
print("RMSE: %.4f" %rmse(y_test, y_pred))
print("RMSLE: %.4f" %rmsle(y_test, y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(y_test, y_pred))


# In[102]:


xgb = xgboost.XGBRegressor(n_threads=4)
xgb = xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)


# In[103]:

print("XGBoost")
print("RMSE: %.4f" %rmse(y_test, y_pred))
print("RMSLE: %.4f" %rmsle(y_test, y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(y_test, y_pred))


# In[116]:


df_importance = pd.DataFrame(x.columns, columns=["Column name"])
df_importance["Feature Importance"] = xgb.feature_importances_
df_importance.sort_values(["Feature Importance"], ascending=False)

