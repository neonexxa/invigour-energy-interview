
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def rmse(actual, predict):
    return sqrt(mean_squared_error(actual, predict))

def rmsle(y, y_pred):
    return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y + 1)).mean())

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def reject_outliers_2(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]


# In[2]:


raw_data = pd.read_csv("train.csv")


# In[3]:


raw_data['pickup_datetime'],raw_data['dropoff_datetime'] = pd.to_datetime(raw_data['pickup_datetime']),pd.to_datetime(raw_data['dropoff_datetime'])


# In[4]:


raw_data.info()


# In[5]:


raw_data.head()


# In[6]:


plt.hist(np.log(raw_data['trip_duration'] + 1), bins=100)


# In[7]:


def get_season(month):
    if month >= 3 and month <= 5:
        return 'spring'
    elif month >= 6 and month <= 8:
        return 'summer'
    elif month >= 9 and month <= 11:
        return 'autumn'
    else:
        return 'winter'


# In[8]:


# format data to month
raw_data['pickup_Month'] = raw_data['pickup_datetime'].map(lambda x: x.month)
raw_data['dropoff_Month'] = raw_data['dropoff_datetime'].map(lambda x: x.month)
raw_data['pickup_Hour'] = raw_data['pickup_datetime'].map(lambda x: x.hour)
raw_data['dropoff_Hour'] = raw_data['dropoff_datetime'].map(lambda x: x.hour)


# In[9]:


raw_data.head()


# In[10]:


raw_data['pickup_season'] = raw_data['pickup_Month'].apply(get_season)
raw_data['dropoff_season'] = raw_data['dropoff_Month'].apply(get_season)


# In[11]:


plt.hist(raw_data['dropoff_season'], bins=100)


# In[12]:


sep_mon = pd.DataFrame(raw_data["pickup_datetime"].dt.month.values, columns=["Month"])
sep_mon["Passenger"] = raw_data["passenger_count"].copy()


# In[13]:


sep_mon.head()


# In[14]:


aggr_sum_mon = sep_mon.groupby("Month").sum()


# In[15]:


aggr_sum_mon.plot(kind='bar')


# In[16]:


raw_data = raw_data.assign(pickup_time=pd.cut(raw_data.pickup_datetime.dt.hour, [-1, 12, 16, 24], labels=['Morning', 'Afternoon', 'Evening']))
raw_data = raw_data.assign(dropoff_time=pd.cut(raw_data.dropoff_datetime.dt.hour, [-1, 12, 16, 24], labels=['Morning', 'Afternoon', 'Evening']))


# In[17]:


raw_data.head()


# In[18]:


new_df = pd.DataFrame(raw_data["pickup_datetime"].dt.hour.values, columns=["Hours"])
new_df["Passenger"] = raw_data["passenger_count"].copy()


# In[19]:


aggr_sum = new_df.groupby("Hours").sum()


# In[20]:


aggr_sum.plot(kind='bar')


# In[21]:


plt.hist(raw_data['dropoff_time'], bins=100)


# In[22]:


# play with lon and lat pulak


# In[23]:


def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


# In[24]:


raw_data.info()


# In[25]:


raw_data['pickup_latitude'][1]


# In[26]:


print( distance((raw_data['pickup_latitude'][1], raw_data['pickup_longitude'][1]), (raw_data['dropoff_latitude'][1], raw_data['dropoff_longitude'][1])) )


# In[27]:


len(raw_data.ix[:])


# In[28]:


# becareful with this part, lame sikit
raw_data['jarak'] = [distance((raw_data['pickup_latitude'][m], raw_data['pickup_longitude'][m]), (raw_data['dropoff_latitude'][m], raw_data['dropoff_longitude'][m])) for m in range(len(raw_data.ix[:]))]


# In[29]:


raw_data.head()


# In[30]:


v = np.array(raw_data['passenger_count'])
w = np.array(raw_data['jarak'])
print(v)


# In[31]:


# linear
mpl.rcParams['agg.path.chunksize'] = 10000
plt.scatter(v, w)
plt.yscale('linear')
plt.title('linear')
plt.show()


# In[32]:


raw_data.head()


# In[33]:


raw_data.corr()


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.matshow(raw_data.corr())


# In[35]:


raw_data.head()


# In[36]:


# after formatting the date and time, there are few more item i need to remove before begin the prediction as such vendor id and id and the target which is the trip duration
# data prep

X = raw_data.drop(['id','vendor_id','pickup_datetime','dropoff_datetime','store_and_fwd_flag','trip_duration','pickup_season','dropoff_season','pickup_time','dropoff_time'],axis=1)
Y = raw_data["trip_duration"]

X.describe()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[37]:


X_train.head()


# In[38]:


import pickle


# In[39]:


def createmodel(modelalgo,clf):
    # Dump the trained decision tree classifier with Pickle
    model_name = modelalgo+'.pkl'
    # Open the file to save as pkl file
    model_name_pkl = open(model_name, 'wb')
    pickle.dump(clf, model_name_pkl)
    # Close the pickle instances
    model_name_pkl.close()


# In[40]:


# begin decision tree regression kat sini

dtrtaxi = tree.DecisionTreeRegressor()
dtrtaxi = dtrtaxi.fit(X_train,y_train)
dtrtaxi_y_pred = dtrtaxi.predict(X_test)
print("Decision Tree Regressor")
print("RMSE: %.4f" %rmse(y_test, dtrtaxi_y_pred))
print("RMSLE: %.4f" %rmsle(y_test, dtrtaxi_y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(y_test, dtrtaxi_y_pred))

#pass to create model function
createmodel('dtrtaxi',dtrtaxi)


# In[41]:


from sklearn.ensemble import RandomForestRegressor
rfrtaxi = RandomForestRegressor()
rfrtaxi.fit(X_train, y_train)
rfrtaxi_y_pred = rfrtaxi.predict(X_test)


# In[42]:


print("Random Forest Regressor")
print("RMSE: %.4f" %rmse(y_test, rfrtaxi_y_pred))
print("RMSLE: %.4f" %rmsle(y_test, rfrtaxi_y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(y_test, rfrtaxi_y_pred))
#pass to create model function
createmodel('rfrtaxi',rfrtaxi)


# In[43]:


# check important features
df_importance = pd.DataFrame(X.columns, columns=["Column name"])
df_importance["Feature Importance DTR"] = dtrtaxi.feature_importances_
df_importance["Feature Importance RFR"] = rfrtaxi.feature_importances_
# df_importance.sort_values(["Feature Importance"], ascending=False)
df_importance


# In[44]:


from sklearn import preprocessing
NewX = raw_data.drop(['id','vendor_id','pickup_datetime','dropoff_datetime','trip_duration'],axis=1)
NewY = raw_data["trip_duration"]

# encode data prep
le_flag = preprocessing.LabelEncoder()
NewX["store_and_fwd_flag"] = le_flag.fit_transform(NewX["store_and_fwd_flag"])

le_pseason = preprocessing.LabelEncoder()
NewX["pickup_season"] = le_pseason.fit_transform(NewX["pickup_season"])

le_dseason = preprocessing.LabelEncoder()
NewX["dropoff_season"] = le_dseason.fit_transform(NewX["dropoff_season"])

le_ptime = preprocessing.LabelEncoder()
NewX["pickup_time"] = le_ptime.fit_transform(NewX["pickup_time"])

le_dtime = preprocessing.LabelEncoder()
NewX["dropoff_time"] = le_dtime.fit_transform(NewX["dropoff_time"])


# In[45]:


NewX


# In[46]:


NewX_train, NewX_test, Newy_train, Newy_test = train_test_split(NewX, NewY, test_size=0.05, random_state=1)


# In[47]:


# begin decision tree regression kat sini for NewX and NewY

Newdtrtaxi = tree.DecisionTreeRegressor()
Newdtrtaxi = Newdtrtaxi.fit(NewX_train,Newy_train)
Newdtrtaxi_y_pred = Newdtrtaxi.predict(NewX_test)
print("Decision Tree Regressor")
print("RMSE: %.4f" %rmse(Newy_test, Newdtrtaxi_y_pred))
print("RMSLE: %.4f" %rmsle(Newy_test, Newdtrtaxi_y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(Newy_test, Newdtrtaxi_y_pred))

#pass to create model function
createmodel('Newdtrtaxi',Newdtrtaxi)


# In[48]:


Newrfrtaxi = RandomForestRegressor()
Newrfrtaxi.fit(NewX_train, Newy_train)
Newrfrtaxi_y_pred = Newrfrtaxi.predict(NewX_test)
print("Random Forest Regressor")
print("RMSE: %.4f" %rmse(Newy_test, Newrfrtaxi_y_pred))
print("RMSLE: %.4f" %rmsle(Newy_test, Newrfrtaxi_y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(Newy_test, Newrfrtaxi_y_pred))
#pass to create model function
createmodel('Newrfrtaxi',Newrfrtaxi)


# In[49]:


from sklearn.ensemble import GradientBoostingRegressor
Newgbrtaxi = RandomForestRegressor()
Newgbrtaxi.fit(NewX_train, Newy_train)
Newgbrtaxi_y_pred = Newgbrtaxi.predict(NewX_test)
print("Gradient Boosting Regressor")
print("RMSE: %.4f" %rmse(Newy_test, Newgbrtaxi_y_pred))
print("RMSLE: %.4f" %rmsle(Newy_test, Newgbrtaxi_y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(Newy_test, Newgbrtaxi_y_pred))
#pass to create model function
createmodel('Newgbrtaxi',Newgbrtaxi)


# In[50]:


# regen model since the features in the test set does not have dropoff datetime column
RegenX = raw_data.drop(['id','vendor_id','pickup_datetime','dropoff_datetime','dropoff_Month','dropoff_Hour','dropoff_season','dropoff_time','trip_duration'],axis=1)
RegenY = raw_data["trip_duration"]

# encode data prep
le_flag = preprocessing.LabelEncoder()
RegenX["store_and_fwd_flag"] = le_flag.fit_transform(RegenX["store_and_fwd_flag"])

le_pseason = preprocessing.LabelEncoder()
RegenX["pickup_season"] = le_pseason.fit_transform(RegenX["pickup_season"])

# le_dseason = preprocessing.LabelEncoder()
# NewX["dropoff_season"] = le_dseason.fit_transform(NewX["dropoff_season"])

le_ptime = preprocessing.LabelEncoder()
RegenX["pickup_time"] = le_ptime.fit_transform(RegenX["pickup_time"])

# le_dtime = preprocessing.LabelEncoder()
# NewX["dropoff_time"] = le_dtime.fit_transform(NewX["dropoff_time"])

RegenX


# In[51]:


# new split data
RegenX_train, RegenX_test, Regeny_train, Regeny_test = train_test_split(RegenX, RegenY, test_size=0.05, random_state=1)

# rerun with gboostReg
Regengbrtaxi = RandomForestRegressor()
Regengbrtaxi.fit(RegenX_train, Regeny_train)


# In[52]:


Regengbrtaxi_y_pred = Regengbrtaxi.predict(RegenX_test)
print("Gradient Boosting Regressor")
print("RMSE: %.4f" %rmse(Regeny_test, Regengbrtaxi_y_pred))
print("RMSLE: %.4f" %rmsle(Regeny_test, Regengbrtaxi_y_pred))
print("MAPE: %.4f" %mean_absolute_percentage_error(Regeny_test, Regengbrtaxi_y_pred))
#pass to create model function
createmodel('Regengbrtaxi',Regengbrtaxi)

