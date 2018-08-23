
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[11]:


all_df=pd.read_csv("../python pracice/titan/train.csv")


# In[12]:


all_df.head()


# In[14]:


all_df=all_df[['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[24]:


all_df.head()


# In[31]:


all_df.isnull().sum()


# In[32]:


age_mean=all_df["Age"].mean()
all_df["Age"]=all_df["Age"].fillna(age_mean)


# In[35]:


all_df["Sex"]=all_df["Sex"].map({"male":1,"female":0}).astype(int)


# In[36]:


df=all_df


# In[37]:


df.Sex.head()


# In[38]:


df=df.drop(["Name"],axis=1)


# In[40]:


df=df.drop(["Embarked"],axis=1)


# In[41]:


df.head()


# In[45]:


ndarray=df.values


# In[47]:


print(type(ndarray))
ndarray.shape


# In[48]:


Label=ndarray[:,0]
Features=ndarray[:,1:]


# In[51]:


Label[0:10]


# In[54]:


print(Features[0:5])
print(Features.shape)


# In[57]:


np.random.seed(10)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[58]:


model=Sequential()


# In[59]:


model.add(Dense(input_dim=6,units=128,
               kernel_initializer="uniform",
               activation="relu"))
model.add(Dropout(0.25))


# In[60]:


model.add(Dense(units=64,
               kernel_initializer="uniform",
               activation="relu"))
model.add(Dropout(0.1))


# In[61]:


model.add(Dense(units=32,
               kernel_initializer="uniform",
               activation="relu"))


# In[62]:


model.add(Dense(units=1,
               kernel_initializer="uniform",
               activation="sigmoid"))


# In[63]:


model.summary()


# In[65]:


model.compile(loss="binary_crossentropy",
            optimizer="adam",metrics=["accuracy"])


# In[68]:


train_history=model.fit(x=Features,y=Label,
                       validation_split=0.1,
                       epochs=200,
                       batch_size=20,verbose=2)


# In[69]:


type(model)


# In[70]:


print(model)


# In[71]:


model


# In[73]:


for_test=pd.read_csv("../python pracice/titan/test.csv")


# In[75]:


for_test.head()


# In[76]:


test=for_test[["Pclass","Sex","Age","SibSp","Parch","Fare"]]


# In[79]:


test.isnull().sum()


# In[84]:


test.isnull().sum()


# In[86]:


fare_mean=test.Fare.mean()


# In[87]:


test["Fare"]=test["Fare"].fillna(fare_mean)


# In[88]:


test.isnull().sum()


# In[90]:


print(fare_mean)


# In[91]:


new_fare_mean=test.Fare.mean()
print(new_fare_mean)


# In[92]:


test.head()


# In[99]:


test["Sex"]=test["Sex"].map({"male":1,"female":0}).astype(int)


# In[100]:


test["Sex"][0:5]


# In[101]:


ndarray_for_test=test.values


# In[102]:


ndarray_for_test[0:5]


# In[103]:


probability=model.predict(ndarray_for_test)


# In[105]:


probability.shape


# In[108]:


output=pd.DataFrame(probability)


# In[109]:


print(output)


# In[110]:


def change_output(x):
    if x>=0.5:
        x=1
    elif x<0.5:
        x=0
    else:
        x="error"
        
    return x


# In[114]:


output[0].apply(change_output)


# In[115]:


result=output[0].apply(change_output)


# In[116]:


result.to_csv("prediction.csv")


# In[118]:


r1=pd.read_csv("../python pracice/prediction.csv")


# In[121]:


r1.describe()

