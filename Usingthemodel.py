
# coding: utf-8

# In[21]:


import pickle
import pandas as pd
import math
from sklearn import preprocessing

def get_season(month):
    if month >= 3 and month <= 5:
        return 'spring'
    elif month >= 6 and month <= 8:
        return 'summer'
    elif month >= 9 and month <= 11:
        return 'autumn'
    else:
        return 'winter'

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


# In[9]:


# load the test set
test_data = pd.read_csv("test.csv")
test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'])
test_data.head()


# In[25]:


test_data['pickup_Month'] = test_data['pickup_datetime'].map(lambda x: x.month)
test_data['pickup_Hour'] = test_data['pickup_datetime'].map(lambda x: x.hour)
test_data = test_data.assign(pickup_time=pd.cut(test_data.pickup_datetime.dt.hour, [-1, 12, 16, 24], labels=['Morning', 'Afternoon', 'Evening']))


# In[12]:


test_data['pickup_season'] = test_data['pickup_Month'].apply(get_season)


# In[17]:


test_data['jarak'] = [distance((test_data['pickup_latitude'][m], test_data['pickup_longitude'][m]), (test_data['dropoff_latitude'][m], test_data['dropoff_longitude'][m])) for m in range(len(test_data.ix[:]))]


# In[24]:


test_data.head()


# In[26]:


X = test_data.drop(['id','vendor_id','pickup_datetime'],axis=1)

# encode data prep
le_flag = preprocessing.LabelEncoder()
X["store_and_fwd_flag"] = le_flag.fit_transform(X["store_and_fwd_flag"])

le_pseason = preprocessing.LabelEncoder()
X["pickup_season"] = le_pseason.fit_transform(X["pickup_season"])

le_ptime = preprocessing.LabelEncoder()
X["pickup_time"] = le_ptime.fit_transform(X["pickup_time"])

# no trip duraction available in the dataset
# Y = raw_data["trip_duration"] 


# In[28]:


model_pkl = open('Regengbrtaxi.pkl', 'rb')
model = pickle.load(model_pkl)
y_pred = model.predict(X)
print ("Loaded model : ", model)
print('Prediction : ',y_pred)


# In[34]:


compiled_result = test_data.drop(['pickup_time','vendor_id','pickup_datetime','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','pickup_Month','pickup_Hour','pickup_season','jarak'],axis=1)


# In[35]:


compiled_result['trip_duration'] = y_pred


# In[36]:


compiled_result


# In[37]:


compiled_result.to_csv('predict.csv')


# In[39]:


df_importance = pd.DataFrame(X.columns, columns=["Column name"])
df_importance["Feature Importance"] = model.feature_importances_


# In[40]:


df_importance

