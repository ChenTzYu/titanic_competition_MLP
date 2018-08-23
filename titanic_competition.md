

```python
import numpy as np
import pandas as pd
```


```python
all_df=pd.read_csv("../python pracice/titan/train.csv")
```


```python
all_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df=all_df[['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']]
```


```python
all_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df.isnull().sum()
```




    Survived      0
    Pclass        0
    Name          0
    Sex           0
    Age         177
    SibSp         0
    Parch         0
    Fare          0
    Embarked      2
    dtype: int64




```python
age_mean=all_df["Age"].mean()
all_df["Age"]=all_df["Age"].fillna(age_mean)
```


```python
all_df["Sex"]=all_df["Sex"].map({"male":1,"female":0}).astype(int)
```


```python
df=all_df
```


```python
df.Sex.head()
```




    0    1
    1    0
    2    0
    3    0
    4    1
    Name: Sex, dtype: int32




```python
df=df.drop(["Name"],axis=1)
```


```python
df=df.drop(["Embarked"],axis=1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
ndarray=df.values
```


```python
print(type(ndarray))
ndarray.shape
```

    <class 'numpy.ndarray'>
    




    (891, 7)




```python
Label=ndarray[:,0]
Features=ndarray[:,1:]
```


```python
Label[0:10]
```




    array([0., 1., 1., 1., 0., 0., 0., 0., 1., 1.])




```python
print(Features[0:5])
print(Features.shape)
```

    [[ 3.      1.     22.      1.      0.      7.25  ]
     [ 1.      0.     38.      1.      0.     71.2833]
     [ 3.      0.     26.      0.      0.      7.925 ]
     [ 1.      0.     35.      1.      0.     53.1   ]
     [ 3.      1.     35.      0.      0.      8.05  ]]
    (891, 6)
    


```python
np.random.seed(10)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

```


```python
model=Sequential()
```


```python
model.add(Dense(input_dim=6,units=128,
               kernel_initializer="uniform",
               activation="relu"))
model.add(Dropout(0.25))
```


```python
model.add(Dense(units=64,
               kernel_initializer="uniform",
               activation="relu"))
model.add(Dropout(0.1))
```


```python
model.add(Dense(units=32,
               kernel_initializer="uniform",
               activation="relu"))

```


```python
model.add(Dense(units=1,
               kernel_initializer="uniform",
               activation="sigmoid"))
```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 128)               896       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 11,265
    Trainable params: 11,265
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss="binary_crossentropy",
            optimizer="adam",metrics=["accuracy"])
```


```python
train_history=model.fit(x=Features,y=Label,
                       validation_split=0.1,
                       epochs=200,
                       batch_size=20,verbose=2)
```

    Train on 801 samples, validate on 90 samples
    Epoch 1/200
     - 0s - loss: 0.4630 - acc: 0.7978 - val_loss: 0.3742 - val_acc: 0.8444
    Epoch 2/200
     - 0s - loss: 0.4601 - acc: 0.7915 - val_loss: 0.3986 - val_acc: 0.8333
    Epoch 3/200
     - 0s - loss: 0.4473 - acc: 0.8040 - val_loss: 0.3656 - val_acc: 0.8444
    Epoch 4/200
     - 0s - loss: 0.4489 - acc: 0.8140 - val_loss: 0.3719 - val_acc: 0.8444
    Epoch 5/200
     - 0s - loss: 0.4579 - acc: 0.7990 - val_loss: 0.3806 - val_acc: 0.8222
    Epoch 6/200
     - 0s - loss: 0.4385 - acc: 0.8127 - val_loss: 0.3719 - val_acc: 0.8556
    Epoch 7/200
     - 0s - loss: 0.4493 - acc: 0.8027 - val_loss: 0.3684 - val_acc: 0.8333
    Epoch 8/200
     - 0s - loss: 0.4412 - acc: 0.8240 - val_loss: 0.3888 - val_acc: 0.8333
    Epoch 9/200
     - 0s - loss: 0.4534 - acc: 0.8027 - val_loss: 0.3691 - val_acc: 0.8333
    Epoch 10/200
     - 0s - loss: 0.4456 - acc: 0.8165 - val_loss: 0.3662 - val_acc: 0.8333
    Epoch 11/200
     - 0s - loss: 0.4423 - acc: 0.8065 - val_loss: 0.3598 - val_acc: 0.8556
    Epoch 12/200
     - 0s - loss: 0.4593 - acc: 0.8077 - val_loss: 0.3629 - val_acc: 0.8222
    Epoch 13/200
     - 0s - loss: 0.4558 - acc: 0.8040 - val_loss: 0.3785 - val_acc: 0.8333
    Epoch 14/200
     - 0s - loss: 0.4608 - acc: 0.8065 - val_loss: 0.3712 - val_acc: 0.8556
    Epoch 15/200
     - 0s - loss: 0.4653 - acc: 0.8140 - val_loss: 0.3621 - val_acc: 0.8444
    Epoch 16/200
     - 0s - loss: 0.4438 - acc: 0.8015 - val_loss: 0.3561 - val_acc: 0.8556
    Epoch 17/200
     - 0s - loss: 0.4467 - acc: 0.7965 - val_loss: 0.3603 - val_acc: 0.8556
    Epoch 18/200
     - 0s - loss: 0.4444 - acc: 0.8015 - val_loss: 0.3538 - val_acc: 0.8556
    Epoch 19/200
     - 0s - loss: 0.4407 - acc: 0.8052 - val_loss: 0.3551 - val_acc: 0.8556
    Epoch 20/200
     - 0s - loss: 0.4311 - acc: 0.8202 - val_loss: 0.3531 - val_acc: 0.8333
    Epoch 21/200
     - 0s - loss: 0.4368 - acc: 0.8077 - val_loss: 0.3639 - val_acc: 0.8333
    Epoch 22/200
     - 0s - loss: 0.4333 - acc: 0.8152 - val_loss: 0.3711 - val_acc: 0.8444
    Epoch 23/200
     - 0s - loss: 0.4403 - acc: 0.8115 - val_loss: 0.3603 - val_acc: 0.8556
    Epoch 24/200
     - 0s - loss: 0.4408 - acc: 0.8090 - val_loss: 0.3512 - val_acc: 0.8556
    Epoch 25/200
     - 0s - loss: 0.4472 - acc: 0.8002 - val_loss: 0.3555 - val_acc: 0.8222
    Epoch 26/200
     - 0s - loss: 0.4279 - acc: 0.8140 - val_loss: 0.3534 - val_acc: 0.8444
    Epoch 27/200
     - 0s - loss: 0.4707 - acc: 0.8065 - val_loss: 0.3605 - val_acc: 0.8556
    Epoch 28/200
     - 0s - loss: 0.4398 - acc: 0.8065 - val_loss: 0.3595 - val_acc: 0.8556
    Epoch 29/200
     - 0s - loss: 0.4384 - acc: 0.8015 - val_loss: 0.3462 - val_acc: 0.8667
    Epoch 30/200
     - 0s - loss: 0.4350 - acc: 0.8102 - val_loss: 0.3490 - val_acc: 0.8667
    Epoch 31/200
     - 0s - loss: 0.4404 - acc: 0.8102 - val_loss: 0.3522 - val_acc: 0.8556
    Epoch 32/200
     - 0s - loss: 0.4294 - acc: 0.8165 - val_loss: 0.3401 - val_acc: 0.8556
    Epoch 33/200
     - 0s - loss: 0.4415 - acc: 0.8077 - val_loss: 0.3390 - val_acc: 0.8556
    Epoch 34/200
     - 0s - loss: 0.4406 - acc: 0.8215 - val_loss: 0.3527 - val_acc: 0.8556
    Epoch 35/200
     - 0s - loss: 0.4485 - acc: 0.8065 - val_loss: 0.3558 - val_acc: 0.8444
    Epoch 36/200
     - 0s - loss: 0.4313 - acc: 0.8065 - val_loss: 0.3375 - val_acc: 0.8444
    Epoch 37/200
     - 0s - loss: 0.4295 - acc: 0.8140 - val_loss: 0.3391 - val_acc: 0.8444
    Epoch 38/200
     - 0s - loss: 0.4294 - acc: 0.8065 - val_loss: 0.3384 - val_acc: 0.8556
    Epoch 39/200
     - 0s - loss: 0.4361 - acc: 0.8165 - val_loss: 0.3362 - val_acc: 0.8667
    Epoch 40/200
     - 0s - loss: 0.4278 - acc: 0.8127 - val_loss: 0.3386 - val_acc: 0.8556
    Epoch 41/200
     - 0s - loss: 0.4346 - acc: 0.8015 - val_loss: 0.3326 - val_acc: 0.8556
    Epoch 42/200
     - 0s - loss: 0.4283 - acc: 0.8077 - val_loss: 0.3365 - val_acc: 0.8556
    Epoch 43/200
     - 0s - loss: 0.4384 - acc: 0.8077 - val_loss: 0.3309 - val_acc: 0.8333
    Epoch 44/200
     - 0s - loss: 0.4383 - acc: 0.8002 - val_loss: 0.3359 - val_acc: 0.8556
    Epoch 45/200
     - 0s - loss: 0.4331 - acc: 0.8102 - val_loss: 0.3532 - val_acc: 0.8556
    Epoch 46/200
     - 0s - loss: 0.4628 - acc: 0.8002 - val_loss: 0.3532 - val_acc: 0.8444
    Epoch 47/200
     - 0s - loss: 0.4307 - acc: 0.8040 - val_loss: 0.3404 - val_acc: 0.8444
    Epoch 48/200
     - 0s - loss: 0.4283 - acc: 0.8140 - val_loss: 0.3413 - val_acc: 0.8444
    Epoch 49/200
     - 0s - loss: 0.4280 - acc: 0.8127 - val_loss: 0.3416 - val_acc: 0.8444
    Epoch 50/200
     - 0s - loss: 0.4253 - acc: 0.8190 - val_loss: 0.3359 - val_acc: 0.8333
    Epoch 51/200
     - 0s - loss: 0.4221 - acc: 0.8077 - val_loss: 0.3402 - val_acc: 0.8444
    Epoch 52/200
     - 0s - loss: 0.4369 - acc: 0.8115 - val_loss: 0.3435 - val_acc: 0.8556
    Epoch 53/200
     - 0s - loss: 0.4549 - acc: 0.8077 - val_loss: 0.3411 - val_acc: 0.8333
    Epoch 54/200
     - 0s - loss: 0.4228 - acc: 0.8177 - val_loss: 0.3436 - val_acc: 0.8667
    Epoch 55/200
     - 0s - loss: 0.4184 - acc: 0.8227 - val_loss: 0.3324 - val_acc: 0.8444
    Epoch 56/200
     - 0s - loss: 0.4280 - acc: 0.8190 - val_loss: 0.3329 - val_acc: 0.8333
    Epoch 57/200
     - 0s - loss: 0.4259 - acc: 0.8165 - val_loss: 0.3337 - val_acc: 0.8444
    Epoch 58/200
     - 0s - loss: 0.4112 - acc: 0.8240 - val_loss: 0.3336 - val_acc: 0.8556
    Epoch 59/200
     - 0s - loss: 0.4232 - acc: 0.8240 - val_loss: 0.3350 - val_acc: 0.8444
    Epoch 60/200
     - 0s - loss: 0.4176 - acc: 0.8152 - val_loss: 0.3287 - val_acc: 0.8667
    Epoch 61/200
     - 0s - loss: 0.4317 - acc: 0.8065 - val_loss: 0.3543 - val_acc: 0.8333
    Epoch 62/200
     - 0s - loss: 0.4355 - acc: 0.8227 - val_loss: 0.3373 - val_acc: 0.8444
    Epoch 63/200
     - 0s - loss: 0.4267 - acc: 0.8252 - val_loss: 0.3356 - val_acc: 0.8556
    Epoch 64/200
     - 0s - loss: 0.4197 - acc: 0.8177 - val_loss: 0.3315 - val_acc: 0.8444
    Epoch 65/200
     - 0s - loss: 0.4270 - acc: 0.8140 - val_loss: 0.3344 - val_acc: 0.8556
    Epoch 66/200
     - 0s - loss: 0.4144 - acc: 0.8165 - val_loss: 0.3458 - val_acc: 0.8333
    Epoch 67/200
     - 0s - loss: 0.4259 - acc: 0.8127 - val_loss: 0.3316 - val_acc: 0.8556
    Epoch 68/200
     - 0s - loss: 0.4142 - acc: 0.8252 - val_loss: 0.3313 - val_acc: 0.8444
    Epoch 69/200
     - 0s - loss: 0.4075 - acc: 0.8277 - val_loss: 0.3359 - val_acc: 0.8556
    Epoch 70/200
     - 0s - loss: 0.4202 - acc: 0.8127 - val_loss: 0.3362 - val_acc: 0.8556
    Epoch 71/200
     - 0s - loss: 0.4262 - acc: 0.8177 - val_loss: 0.3389 - val_acc: 0.8444
    Epoch 72/200
     - 0s - loss: 0.4257 - acc: 0.8152 - val_loss: 0.3357 - val_acc: 0.8444
    Epoch 73/200
     - 0s - loss: 0.4222 - acc: 0.8215 - val_loss: 0.3390 - val_acc: 0.8444
    Epoch 74/200
     - 0s - loss: 0.4327 - acc: 0.8140 - val_loss: 0.3312 - val_acc: 0.8444
    Epoch 75/200
     - 0s - loss: 0.4325 - acc: 0.8127 - val_loss: 0.3317 - val_acc: 0.8444
    Epoch 76/200
     - 0s - loss: 0.4126 - acc: 0.8202 - val_loss: 0.3308 - val_acc: 0.8333
    Epoch 77/200
     - 0s - loss: 0.4344 - acc: 0.8102 - val_loss: 0.3377 - val_acc: 0.8556
    Epoch 78/200
     - 0s - loss: 0.4232 - acc: 0.8140 - val_loss: 0.3364 - val_acc: 0.8444
    Epoch 79/200
     - 0s - loss: 0.4262 - acc: 0.8027 - val_loss: 0.3253 - val_acc: 0.8444
    Epoch 80/200
     - 0s - loss: 0.4303 - acc: 0.8115 - val_loss: 0.3278 - val_acc: 0.8333
    Epoch 81/200
     - 0s - loss: 0.4457 - acc: 0.8002 - val_loss: 0.3326 - val_acc: 0.8556
    Epoch 82/200
     - 0s - loss: 0.4273 - acc: 0.8115 - val_loss: 0.3451 - val_acc: 0.8667
    Epoch 83/200
     - 0s - loss: 0.4224 - acc: 0.8190 - val_loss: 0.3275 - val_acc: 0.8667
    Epoch 84/200
     - 0s - loss: 0.4095 - acc: 0.8215 - val_loss: 0.3327 - val_acc: 0.8222
    Epoch 85/200
     - 0s - loss: 0.4284 - acc: 0.8140 - val_loss: 0.3383 - val_acc: 0.8556
    Epoch 86/200
     - 0s - loss: 0.4150 - acc: 0.8165 - val_loss: 0.3202 - val_acc: 0.8444
    Epoch 87/200
     - 0s - loss: 0.4108 - acc: 0.8240 - val_loss: 0.3217 - val_acc: 0.8667
    Epoch 88/200
     - 0s - loss: 0.4155 - acc: 0.8277 - val_loss: 0.3244 - val_acc: 0.8556
    Epoch 89/200
     - 0s - loss: 0.4133 - acc: 0.8240 - val_loss: 0.3265 - val_acc: 0.8556
    Epoch 90/200
     - 0s - loss: 0.4303 - acc: 0.8165 - val_loss: 0.3274 - val_acc: 0.8556
    Epoch 91/200
     - 0s - loss: 0.4065 - acc: 0.8277 - val_loss: 0.3332 - val_acc: 0.8556
    Epoch 92/200
     - 0s - loss: 0.4106 - acc: 0.8265 - val_loss: 0.3222 - val_acc: 0.8556
    Epoch 93/200
     - 0s - loss: 0.4262 - acc: 0.8152 - val_loss: 0.3319 - val_acc: 0.8444
    Epoch 94/200
     - 0s - loss: 0.4606 - acc: 0.7878 - val_loss: 0.3592 - val_acc: 0.8222
    Epoch 95/200
     - 0s - loss: 0.4327 - acc: 0.8027 - val_loss: 0.3486 - val_acc: 0.8222
    Epoch 96/200
     - 0s - loss: 0.4310 - acc: 0.8177 - val_loss: 0.3425 - val_acc: 0.8667
    Epoch 97/200
     - 0s - loss: 0.4418 - acc: 0.8090 - val_loss: 0.3360 - val_acc: 0.8556
    Epoch 98/200
     - 0s - loss: 0.4231 - acc: 0.8202 - val_loss: 0.3353 - val_acc: 0.8667
    Epoch 99/200
     - 0s - loss: 0.4231 - acc: 0.8027 - val_loss: 0.3265 - val_acc: 0.8667
    Epoch 100/200
     - 0s - loss: 0.4199 - acc: 0.8240 - val_loss: 0.3281 - val_acc: 0.8556
    Epoch 101/200
     - 0s - loss: 0.4223 - acc: 0.8140 - val_loss: 0.3384 - val_acc: 0.8556
    Epoch 102/200
     - 0s - loss: 0.4382 - acc: 0.8127 - val_loss: 0.3542 - val_acc: 0.8444
    Epoch 103/200
     - 0s - loss: 0.4302 - acc: 0.8152 - val_loss: 0.3382 - val_acc: 0.8222
    Epoch 104/200
     - 0s - loss: 0.4205 - acc: 0.8215 - val_loss: 0.3244 - val_acc: 0.8667
    Epoch 105/200
     - 0s - loss: 0.4196 - acc: 0.8252 - val_loss: 0.3213 - val_acc: 0.8667
    Epoch 106/200
     - 0s - loss: 0.4330 - acc: 0.8152 - val_loss: 0.3281 - val_acc: 0.8667
    Epoch 107/200
     - 0s - loss: 0.4147 - acc: 0.8190 - val_loss: 0.3231 - val_acc: 0.8667
    Epoch 108/200
     - 0s - loss: 0.4101 - acc: 0.8252 - val_loss: 0.3372 - val_acc: 0.8556
    Epoch 109/200
     - 0s - loss: 0.4264 - acc: 0.8127 - val_loss: 0.3236 - val_acc: 0.8667
    Epoch 110/200
     - 0s - loss: 0.4206 - acc: 0.8152 - val_loss: 0.3275 - val_acc: 0.8556
    Epoch 111/200
     - 0s - loss: 0.4080 - acc: 0.8165 - val_loss: 0.3211 - val_acc: 0.8556
    Epoch 112/200
     - 0s - loss: 0.4244 - acc: 0.8040 - val_loss: 0.3296 - val_acc: 0.8333
    Epoch 113/200
     - 0s - loss: 0.4139 - acc: 0.8265 - val_loss: 0.3203 - val_acc: 0.8667
    Epoch 114/200
     - 0s - loss: 0.4086 - acc: 0.8277 - val_loss: 0.3225 - val_acc: 0.8667
    Epoch 115/200
     - 0s - loss: 0.4225 - acc: 0.8152 - val_loss: 0.3274 - val_acc: 0.8667
    Epoch 116/200
     - 0s - loss: 0.4089 - acc: 0.8315 - val_loss: 0.3350 - val_acc: 0.8444
    Epoch 117/200
     - 0s - loss: 0.4063 - acc: 0.8127 - val_loss: 0.3201 - val_acc: 0.8444
    Epoch 118/200
     - 0s - loss: 0.4101 - acc: 0.8252 - val_loss: 0.3211 - val_acc: 0.8556
    Epoch 119/200
     - 0s - loss: 0.4106 - acc: 0.8252 - val_loss: 0.3216 - val_acc: 0.8556
    Epoch 120/200
     - 0s - loss: 0.4078 - acc: 0.8227 - val_loss: 0.3350 - val_acc: 0.8556
    Epoch 121/200
     - 0s - loss: 0.4072 - acc: 0.8302 - val_loss: 0.3239 - val_acc: 0.8667
    Epoch 122/200
     - 0s - loss: 0.4160 - acc: 0.8215 - val_loss: 0.3243 - val_acc: 0.8444
    Epoch 123/200
     - 0s - loss: 0.4164 - acc: 0.8240 - val_loss: 0.3324 - val_acc: 0.8444
    Epoch 124/200
     - 0s - loss: 0.4105 - acc: 0.8227 - val_loss: 0.3303 - val_acc: 0.8333
    Epoch 125/200
     - 0s - loss: 0.4171 - acc: 0.8252 - val_loss: 0.3354 - val_acc: 0.8333
    Epoch 126/200
     - 0s - loss: 0.4100 - acc: 0.8227 - val_loss: 0.3273 - val_acc: 0.8444
    Epoch 127/200
     - 0s - loss: 0.4123 - acc: 0.8265 - val_loss: 0.3185 - val_acc: 0.8556
    Epoch 128/200
     - 0s - loss: 0.4046 - acc: 0.8277 - val_loss: 0.3232 - val_acc: 0.8556
    Epoch 129/200
     - 0s - loss: 0.4049 - acc: 0.8227 - val_loss: 0.3293 - val_acc: 0.8333
    Epoch 130/200
     - 0s - loss: 0.4046 - acc: 0.8190 - val_loss: 0.3335 - val_acc: 0.8667
    Epoch 131/200
     - 0s - loss: 0.4172 - acc: 0.8265 - val_loss: 0.3293 - val_acc: 0.8444
    Epoch 132/200
     - 0s - loss: 0.4125 - acc: 0.8240 - val_loss: 0.3184 - val_acc: 0.8556
    Epoch 133/200
     - 0s - loss: 0.4102 - acc: 0.8265 - val_loss: 0.3263 - val_acc: 0.8444
    Epoch 134/200
     - 0s - loss: 0.4208 - acc: 0.8065 - val_loss: 0.3387 - val_acc: 0.8444
    Epoch 135/200
     - 0s - loss: 0.4215 - acc: 0.8115 - val_loss: 0.3320 - val_acc: 0.8556
    Epoch 136/200
     - 0s - loss: 0.4244 - acc: 0.8165 - val_loss: 0.3300 - val_acc: 0.8556
    Epoch 137/200
     - 0s - loss: 0.4191 - acc: 0.8165 - val_loss: 0.3326 - val_acc: 0.8333
    Epoch 138/200
     - 0s - loss: 0.4177 - acc: 0.8215 - val_loss: 0.3300 - val_acc: 0.8667
    Epoch 139/200
     - 0s - loss: 0.4108 - acc: 0.8290 - val_loss: 0.3334 - val_acc: 0.8333
    Epoch 140/200
     - 0s - loss: 0.4229 - acc: 0.8040 - val_loss: 0.3236 - val_acc: 0.8444
    Epoch 141/200
     - 0s - loss: 0.4124 - acc: 0.8165 - val_loss: 0.3187 - val_acc: 0.8556
    Epoch 142/200
     - 0s - loss: 0.4113 - acc: 0.8252 - val_loss: 0.3281 - val_acc: 0.8556
    Epoch 143/200
     - 0s - loss: 0.4146 - acc: 0.8302 - val_loss: 0.3388 - val_acc: 0.8444
    Epoch 144/200
     - 0s - loss: 0.4230 - acc: 0.8140 - val_loss: 0.3327 - val_acc: 0.8444
    Epoch 145/200
     - 0s - loss: 0.4154 - acc: 0.8202 - val_loss: 0.3296 - val_acc: 0.8444
    Epoch 146/200
     - 0s - loss: 0.4131 - acc: 0.8240 - val_loss: 0.3288 - val_acc: 0.8444
    Epoch 147/200
     - 0s - loss: 0.4042 - acc: 0.8315 - val_loss: 0.3249 - val_acc: 0.8778
    Epoch 148/200
     - 0s - loss: 0.4059 - acc: 0.8240 - val_loss: 0.3252 - val_acc: 0.8556
    Epoch 149/200
     - 0s - loss: 0.4082 - acc: 0.8302 - val_loss: 0.3290 - val_acc: 0.8556
    Epoch 150/200
     - 0s - loss: 0.4194 - acc: 0.8240 - val_loss: 0.3248 - val_acc: 0.8444
    Epoch 151/200
     - 0s - loss: 0.4051 - acc: 0.8240 - val_loss: 0.3215 - val_acc: 0.8444
    Epoch 152/200
     - 0s - loss: 0.4062 - acc: 0.8315 - val_loss: 0.3174 - val_acc: 0.8556
    Epoch 153/200
     - 0s - loss: 0.4175 - acc: 0.8177 - val_loss: 0.3202 - val_acc: 0.8556
    Epoch 154/200
     - 0s - loss: 0.3999 - acc: 0.8177 - val_loss: 0.3178 - val_acc: 0.8444
    Epoch 155/200
     - 0s - loss: 0.4032 - acc: 0.8252 - val_loss: 0.3188 - val_acc: 0.8444
    Epoch 156/200
     - 0s - loss: 0.4278 - acc: 0.8140 - val_loss: 0.3273 - val_acc: 0.8556
    Epoch 157/200
     - 0s - loss: 0.4213 - acc: 0.8177 - val_loss: 0.3276 - val_acc: 0.8556
    Epoch 158/200
     - 0s - loss: 0.4353 - acc: 0.8102 - val_loss: 0.3237 - val_acc: 0.8556
    Epoch 159/200
     - 0s - loss: 0.4083 - acc: 0.8177 - val_loss: 0.3205 - val_acc: 0.8556
    Epoch 160/200
     - 0s - loss: 0.4028 - acc: 0.8202 - val_loss: 0.3285 - val_acc: 0.8444
    Epoch 161/200
     - 0s - loss: 0.4144 - acc: 0.8140 - val_loss: 0.3349 - val_acc: 0.8333
    Epoch 162/200
     - 0s - loss: 0.4132 - acc: 0.8177 - val_loss: 0.3251 - val_acc: 0.8667
    Epoch 163/200
     - 0s - loss: 0.4136 - acc: 0.8177 - val_loss: 0.3254 - val_acc: 0.8444
    Epoch 164/200
     - 0s - loss: 0.4104 - acc: 0.8265 - val_loss: 0.3214 - val_acc: 0.8556
    Epoch 165/200
     - 0s - loss: 0.4054 - acc: 0.8252 - val_loss: 0.3304 - val_acc: 0.8444
    Epoch 166/200
     - 0s - loss: 0.4074 - acc: 0.8215 - val_loss: 0.3247 - val_acc: 0.8444
    Epoch 167/200
     - 0s - loss: 0.4281 - acc: 0.8065 - val_loss: 0.3268 - val_acc: 0.8556
    Epoch 168/200
     - 0s - loss: 0.4074 - acc: 0.8190 - val_loss: 0.3244 - val_acc: 0.8444
    Epoch 169/200
     - 0s - loss: 0.4155 - acc: 0.8140 - val_loss: 0.3212 - val_acc: 0.8444
    Epoch 170/200
     - 0s - loss: 0.4111 - acc: 0.8165 - val_loss: 0.3236 - val_acc: 0.8556
    Epoch 171/200
     - 0s - loss: 0.4079 - acc: 0.8227 - val_loss: 0.3214 - val_acc: 0.8444
    Epoch 172/200
     - 0s - loss: 0.4150 - acc: 0.8152 - val_loss: 0.3433 - val_acc: 0.8333
    Epoch 173/200
     - 0s - loss: 0.4198 - acc: 0.8115 - val_loss: 0.3227 - val_acc: 0.8333
    Epoch 174/200
     - 0s - loss: 0.4054 - acc: 0.8302 - val_loss: 0.3227 - val_acc: 0.8444
    Epoch 175/200
     - 0s - loss: 0.4126 - acc: 0.8240 - val_loss: 0.3176 - val_acc: 0.8778
    Epoch 176/200
     - 0s - loss: 0.4181 - acc: 0.8277 - val_loss: 0.3231 - val_acc: 0.8444
    Epoch 177/200
     - 0s - loss: 0.4124 - acc: 0.8252 - val_loss: 0.3208 - val_acc: 0.8444
    Epoch 178/200
     - 0s - loss: 0.4119 - acc: 0.8252 - val_loss: 0.3174 - val_acc: 0.8556
    Epoch 179/200
     - 0s - loss: 0.4021 - acc: 0.8277 - val_loss: 0.3299 - val_acc: 0.8444
    Epoch 180/200
     - 0s - loss: 0.4250 - acc: 0.8065 - val_loss: 0.3243 - val_acc: 0.8556
    Epoch 181/200
     - 0s - loss: 0.4115 - acc: 0.8290 - val_loss: 0.3217 - val_acc: 0.8667
    Epoch 182/200
     - 0s - loss: 0.4169 - acc: 0.8177 - val_loss: 0.3257 - val_acc: 0.8556
    Epoch 183/200
     - 0s - loss: 0.4023 - acc: 0.8265 - val_loss: 0.3315 - val_acc: 0.8556
    Epoch 184/200
     - 0s - loss: 0.3935 - acc: 0.8252 - val_loss: 0.3259 - val_acc: 0.8444
    Epoch 185/200
     - 0s - loss: 0.4039 - acc: 0.8265 - val_loss: 0.3319 - val_acc: 0.8444
    Epoch 186/200
     - 0s - loss: 0.4004 - acc: 0.8215 - val_loss: 0.3154 - val_acc: 0.8556
    Epoch 187/200
     - 0s - loss: 0.4079 - acc: 0.8302 - val_loss: 0.3342 - val_acc: 0.8556
    Epoch 188/200
     - 0s - loss: 0.4085 - acc: 0.8177 - val_loss: 0.3250 - val_acc: 0.8444
    Epoch 189/200
     - 0s - loss: 0.3988 - acc: 0.8277 - val_loss: 0.3285 - val_acc: 0.8556
    Epoch 190/200
     - 0s - loss: 0.4041 - acc: 0.8265 - val_loss: 0.3271 - val_acc: 0.8444
    Epoch 191/200
     - 0s - loss: 0.4090 - acc: 0.8177 - val_loss: 0.3288 - val_acc: 0.8444
    Epoch 192/200
     - 0s - loss: 0.4018 - acc: 0.8265 - val_loss: 0.3228 - val_acc: 0.8222
    Epoch 193/200
     - 0s - loss: 0.4025 - acc: 0.8290 - val_loss: 0.3265 - val_acc: 0.8444
    Epoch 194/200
     - 0s - loss: 0.4074 - acc: 0.8127 - val_loss: 0.3264 - val_acc: 0.8444
    Epoch 195/200
     - 0s - loss: 0.3991 - acc: 0.8327 - val_loss: 0.3241 - val_acc: 0.8444
    Epoch 196/200
     - 0s - loss: 0.4000 - acc: 0.8202 - val_loss: 0.3216 - val_acc: 0.8444
    Epoch 197/200
     - 0s - loss: 0.4148 - acc: 0.8140 - val_loss: 0.3281 - val_acc: 0.8556
    Epoch 198/200
     - 0s - loss: 0.4054 - acc: 0.8202 - val_loss: 0.3307 - val_acc: 0.8444
    Epoch 199/200
     - 0s - loss: 0.4097 - acc: 0.8227 - val_loss: 0.3258 - val_acc: 0.8333
    Epoch 200/200
     - 0s - loss: 0.4039 - acc: 0.8252 - val_loss: 0.3100 - val_acc: 0.8556
    


```python
type(model)
```




    keras.engine.sequential.Sequential




```python
print(model)
```

    <keras.engine.sequential.Sequential object at 0x00000294A887A8D0>
    


```python
model
```




    <keras.engine.sequential.Sequential at 0x294a887a8d0>




```python
for_test=pd.read_csv("../python pracice/titan/test.csv")
```


```python
for_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test=for_test[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
```


```python
test.isnull().sum()
```




    Pclass     0
    Sex        0
    Age       86
    SibSp      0
    Parch      0
    Fare       1
    dtype: int64




```python
test.isnull().sum()
```




    Pclass    0
    Sex       0
    Age       0
    SibSp     0
    Parch     0
    Fare      1
    dtype: int64




```python
fare_mean=test.Fare.mean()
```


```python
test["Fare"]=test["Fare"].fillna(fare_mean)
```

    C:\Anaconda\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    


```python
test.isnull().sum()
```




    Pclass    0
    Sex       0
    Age       0
    SibSp     0
    Parch     0
    Fare      0
    dtype: int64




```python
print(fare_mean)
```

    35.6271884892086
    


```python
new_fare_mean=test.Fare.mean()
print(new_fare_mean)
```

    35.6271884892086
    


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
    </tr>
  </tbody>
</table>
</div>




```python
test["Sex"]=test["Sex"].map({"male":1,"female":0}).astype(int)
```

    C:\Anaconda\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    


```python
test["Sex"][0:5]
```




    0    1
    1    0
    2    1
    3    1
    4    0
    Name: Sex, dtype: int32




```python
ndarray_for_test=test.values
```


```python
ndarray_for_test[0:5]
```




    array([[ 3.    ,  1.    , 34.5   ,  0.    ,  0.    ,  7.8292],
           [ 3.    ,  0.    , 47.    ,  1.    ,  0.    ,  7.    ],
           [ 2.    ,  1.    , 62.    ,  0.    ,  0.    ,  9.6875],
           [ 3.    ,  1.    , 27.    ,  0.    ,  0.    ,  8.6625],
           [ 3.    ,  0.    , 22.    ,  1.    ,  1.    , 12.2875]])




```python
probability=model.predict(ndarray_for_test)
```


```python
probability.shape
```




    (418, 1)




```python
output=pd.DataFrame(probability)
```


```python
print(output)
```

                0
    0    0.105708
    1    0.441802
    2    0.088280
    3    0.124370
    4    0.421544
    5    0.161340
    6    0.586718
    7    0.131567
    8    0.578051
    9    0.119095
    10   0.114556
    11   0.373770
    12   0.999342
    13   0.109483
    14   0.999615
    15   0.995992
    16   0.138348
    17   0.134720
    18   0.424990
    19   0.570232
    20   0.432900
    21   0.347570
    22   0.999309
    23   0.480744
    24   0.915700
    25   0.080777
    26   0.998171
    27   0.130900
    28   0.394281
    29   0.122959
    ..        ...
    388  0.136467
    389  0.110953
    390  0.364012
    391  0.993200
    392  0.329944
    393  0.112666
    394  0.096190
    395  0.998847
    396  0.127252
    397  0.999224
    398  0.133965
    399  0.112538
    400  0.958965
    401  0.124249
    402  0.998119
    403  0.486490
    404  0.279434
    405  0.171200
    406  0.137504
    407  0.317623
    408  0.589534
    409  0.956228
    410  0.590399
    411  0.999625
    412  0.590455
    413  0.115103
    414  0.999773
    415  0.096847
    416  0.115103
    417  0.133016
    
    [418 rows x 1 columns]
    


```python
def change_output(x):
    if x>=0.5:
        x=1
    elif x<0.5:
        x=0
    else:
        x="error"
        
    return x
```


```python
output[0].apply(change_output)
```




    0      0
    1      0
    2      0
    3      0
    4      0
    5      0
    6      1
    7      0
    8      1
    9      0
    10     0
    11     0
    12     1
    13     0
    14     1
    15     1
    16     0
    17     0
    18     0
    19     1
    20     0
    21     0
    22     1
    23     0
    24     1
    25     0
    26     1
    27     0
    28     0
    29     0
          ..
    388    0
    389    0
    390    0
    391    1
    392    0
    393    0
    394    0
    395    1
    396    0
    397    1
    398    0
    399    0
    400    1
    401    0
    402    1
    403    0
    404    0
    405    0
    406    0
    407    0
    408    1
    409    1
    410    1
    411    1
    412    1
    413    0
    414    1
    415    0
    416    0
    417    0
    Name: 0, Length: 418, dtype: int64




```python
result=output[0].apply(change_output)
```


```python
result.to_csv("prediction.csv")
```


```python
r1=pd.read_csv("../python pracice/prediction.csv")
```


```python
r1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>0.342105</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.474983</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


