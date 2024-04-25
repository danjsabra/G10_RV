```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import tensorflow as tf
import keras as keras
plt.style.use('ggplot')
```


```python
codes = {}
codes['US'] = 'USGG10YR'
codes['Germany'] = 'GDBR10'
codes['UK'] = 'GUKG10'
codes['France'] = 'GFRN10'
codes['Australia'] = 'GACGB10'
codes['Canada'] = 'GCAN10YR'
codes['New Zealand'] = 'GNZGB10'
codes['Japan'] = 'JGBS10'
codes['Switzerland'] = 'GSWISS10'
codes['Norway'] = 'GNOR10YR'
codes['Italy'] = 'GBTPGR10'

sheet_names = pd.ExcelFile('G10_RV.xlsx').sheet_names[:11]
dfs = {x: pd.read_excel('G10_RV.xlsx', sheet_name=x)[['Date', 'Last Price']].rename(columns={'Last Price': x}) for x in sheet_names}
df = pd.DataFrame({'Date': dfs[sheet_names[0]]['Date']})  

for key in dfs:
    df = pd.merge(df, dfs[key], on='Date', how='outer')

df = df.set_index('Date').resample('D').asfreq().ffill().dropna()
df.iloc[[0,-1],:]
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
      <th>USGG10YR</th>
      <th>GDBR10</th>
      <th>GUKG10</th>
      <th>GFRN10</th>
      <th>GACGB10</th>
      <th>GCAN10YR</th>
      <th>GNZGB10</th>
      <th>JGBS10</th>
      <th>GSWISS10</th>
      <th>GNOR10YR</th>
      <th>GBTPGR10</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-05</th>
      <td>3.7608</td>
      <td>3.373</td>
      <td>4.016</td>
      <td>3.592</td>
      <td>5.621</td>
      <td>3.565</td>
      <td>5.843</td>
      <td>1.329</td>
      <td>2.009</td>
      <td>4.147</td>
      <td>4.101</td>
    </tr>
    <tr>
      <th>2024-04-12</th>
      <td>4.5216</td>
      <td>2.359</td>
      <td>4.137</td>
      <td>2.865</td>
      <td>4.266</td>
      <td>3.649</td>
      <td>4.839</td>
      <td>0.864</td>
      <td>0.739</td>
      <td>3.707</td>
      <td>3.762</td>
    </tr>
  </tbody>
</table>
</div>




```python
def predictor(method, target, t):

    data = df.copy()
    target = codes[target]
    target_t = f'{target}_{t}'
    ts = [1,5,10,25,50,100]

    for x in data:
        for z in ts:
            data[f'{x}_{z}'] = df[x].diff(z)

    data = data.dropna()
    cutoff = '2020-1-1'
    training = data[data.index < cutoff]
    testing = data[data.index > cutoff]

    if method == 'simple' or method == 'multi':
        if method == 'simple':
            training_X = training[[x for x in training if x.endswith(f'_{t}') and not x.startswith(target)]]
            training_y = training[target_t]
            testing_X = testing[[x for x in testing if x.endswith(f'_{t}') and not x.startswith(target)]]
            testing_y = testing[target_t]
        if method == 'multi':
            training_X = training[[x for x in training if '_' in x and x != target_t]]
            training_y = training[target_t]
            testing_X = testing[[x for x in testing if '_' in x and x != target_t]]
            testing_y = testing[target_t]
    
        model = LinearRegression()
        model.fit(training_X, training_y)
        training_prediction = model.predict(training_X)
        testing_prediction = model.predict(testing_X)
        training_accuracy = round(r2_score(training_y, training_prediction), 2)
        testing_accuracy = round(r2_score(testing_y, testing_prediction), 2)
        prediction = pd.DataFrame(testing[target].copy())
        prediction['c_prediction'] = testing_prediction
        prediction['prediction'] = prediction[target].shift(t) + prediction['c_prediction']
        prediction = prediction[[target, 'prediction']].dropna()
        return prediction, training_accuracy, testing_accuracy
    
    elif method == 'neural':
        training_X = training[[x for x in training if '_' in x and x != target_t]]
        training_y = training[target_t]
        testing_X = testing[[x for x in testing if '_' in x and x != target_t]]
        testing_y = testing[target_t]
        scaler = StandardScaler()
        training_X = scaler.fit_transform(training_X)
        testing_X = scaler.transform(testing_X)
        model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=[training_X.shape[1]]),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        model.fit(training_X, training_y, epochs=200, validation_split=0.2, callbacks=[early_stopping])
        training_prediction = model.predict(training_X)
        testing_prediction = model.predict(testing_X)
        training_accuracy = round(r2_score(training_y, training_prediction), 2)
        testing_accuracy = round(r2_score(testing_y, testing_prediction), 2)
        prediction = pd.DataFrame(testing[target].copy())
        prediction['c_prediction'] = testing_prediction
        prediction['prediction'] = prediction[target].shift(t) + prediction['c_prediction']
        prediction = prediction[[target, 'prediction']].dropna()
        return prediction, training_accuracy, testing_accuracy

def performance(t):
    performances = pd.DataFrame()
    performances['Country'] = codes.keys()
    performances['r2: Simple'] = performances['Country'].apply(lambda x: predictor('simple', x, t)[2])
    performances['r2: Multi'] = performances['Country'].apply(lambda x: predictor('multi', x, t)[2])
    performances['r2: Neural'] = performances['Country'].apply(lambda x: predictor('neural', x, t)[2])
    performances = performances.set_index('Country')
    simple_mean = round(performances['r2: Simple'].mean(), 2)
    multi_mean = round(performances['r2: Multi'].mean(), 2)
    nn_mean = round(performances['r2: Neural'].mean(), 2)
    performances.loc['Mean'] = {'r2: Simple': simple_mean, 'r2: Multi': multi_mean, 'r2: Neural': nn_mean}
    return performances

performance(10)
```

    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.7040 - val_loss: 0.0261
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 542us/step - loss: 0.1039 - val_loss: 0.0110
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - loss: 0.0537 - val_loss: 0.0065
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0312 - val_loss: 0.0063
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0236 - val_loss: 0.0054
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0178 - val_loss: 0.0048
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0150 - val_loss: 0.0045
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0136 - val_loss: 0.0042
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0104 - val_loss: 0.0036
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0103 - val_loss: 0.0036
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0093 - val_loss: 0.0034
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0084 - val_loss: 0.0031
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0077 - val_loss: 0.0029
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0076 - val_loss: 0.0030
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0073 - val_loss: 0.0026
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0073 - val_loss: 0.0025
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 545us/step - loss: 0.0060 - val_loss: 0.0025
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 555us/step - loss: 0.0068 - val_loss: 0.0024
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0058 - val_loss: 0.0022
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0059 - val_loss: 0.0023
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0061 - val_loss: 0.0021
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0058 - val_loss: 0.0019
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0055 - val_loss: 0.0019
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0052 - val_loss: 0.0020
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0053 - val_loss: 0.0019
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0053 - val_loss: 0.0019
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0043 - val_loss: 0.0017
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0045 - val_loss: 0.0019
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0053 - val_loss: 0.0018
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0047 - val_loss: 0.0016
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0045 - val_loss: 0.0016
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0044 - val_loss: 0.0017
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0042 - val_loss: 0.0015
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0042 - val_loss: 0.0018
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0039 - val_loss: 0.0015
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0041 - val_loss: 0.0017
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0039 - val_loss: 0.0015
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0040 - val_loss: 0.0016
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 547us/step - loss: 0.0038 - val_loss: 0.0016
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0040 - val_loss: 0.0016
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0040 - val_loss: 0.0014
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0036 - val_loss: 0.0014
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 537us/step - loss: 0.0036 - val_loss: 0.0016
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0036 - val_loss: 0.0014
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0034 - val_loss: 0.0015
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0035 - val_loss: 0.0016
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0036 - val_loss: 0.0013
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0034 - val_loss: 0.0013
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0034 - val_loss: 0.0016
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0034 - val_loss: 0.0016
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0032 - val_loss: 0.0014
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0034 - val_loss: 0.0015
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0034 - val_loss: 0.0013
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0034 - val_loss: 0.0014
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0033 - val_loss: 0.0015
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0032 - val_loss: 0.0014
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0033 - val_loss: 0.0013
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0033 - val_loss: 0.0013
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0032 - val_loss: 0.0014
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0031 - val_loss: 0.0014
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0030 - val_loss: 0.0013
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0031 - val_loss: 0.0012
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0030 - val_loss: 0.0012
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0027 - val_loss: 0.0012
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0031 - val_loss: 0.0013
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0027 - val_loss: 0.0012
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0028 - val_loss: 0.0012
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0027 - val_loss: 0.0012
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0027 - val_loss: 0.0014
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0029 - val_loss: 0.0011
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0028 - val_loss: 0.0012
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0029 - val_loss: 0.0013
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0026 - val_loss: 0.0011
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0027 - val_loss: 0.0011
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 0.0014
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0026 - val_loss: 0.0012
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0026 - val_loss: 0.0012
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0027 - val_loss: 0.0012
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0027 - val_loss: 0.0011
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0024 - val_loss: 0.0011
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0026 - val_loss: 0.0012
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0026 - val_loss: 0.0011
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0025 - val_loss: 0.0014
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0027 - val_loss: 0.0012
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0026 - val_loss: 0.0012
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0025 - val_loss: 0.0012
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0025 - val_loss: 0.0011
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0022 - val_loss: 0.0012
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0024 - val_loss: 0.0012
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0024 - val_loss: 0.0018
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0023 - val_loss: 0.0012
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0024 - val_loss: 0.0015
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0024 - val_loss: 0.0011
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0023 - val_loss: 0.0013
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0024 - val_loss: 0.0011
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0024 - val_loss: 0.0013
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0023 - val_loss: 0.0012
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0024 - val_loss: 0.0012
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0021 - val_loss: 0.0013
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0025 - val_loss: 0.0011
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0024 - val_loss: 0.0014
    Epoch 103/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0023 - val_loss: 0.0012
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 343us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4341 - val_loss: 0.0136
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - loss: 0.0732 - val_loss: 0.0062
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0352 - val_loss: 0.0035
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0228 - val_loss: 0.0028
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0165 - val_loss: 0.0024
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0128 - val_loss: 0.0021
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0104 - val_loss: 0.0018
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0101 - val_loss: 0.0017
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0086 - val_loss: 0.0016
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0072 - val_loss: 0.0013
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0070 - val_loss: 0.0014
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0061 - val_loss: 0.0012
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0064 - val_loss: 0.0012
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0060 - val_loss: 0.0011
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0059 - val_loss: 0.0011
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0051 - val_loss: 0.0011
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0048 - val_loss: 9.6429e-04
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0048 - val_loss: 9.1691e-04
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0048 - val_loss: 8.3021e-04
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0046 - val_loss: 8.3054e-04
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0046 - val_loss: 8.6338e-04
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0043 - val_loss: 7.6672e-04
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0044 - val_loss: 8.4164e-04
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0040 - val_loss: 7.4468e-04
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0040 - val_loss: 8.4671e-04
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0039 - val_loss: 7.4017e-04
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0038 - val_loss: 7.2054e-04
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0036 - val_loss: 7.1371e-04
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0034 - val_loss: 7.0231e-04
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0037 - val_loss: 6.7125e-04
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0037 - val_loss: 7.1220e-04
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0037 - val_loss: 6.7811e-04
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0036 - val_loss: 6.3656e-04
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0032 - val_loss: 6.7812e-04
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0035 - val_loss: 5.8461e-04
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0032 - val_loss: 6.2677e-04
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0031 - val_loss: 5.7135e-04
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0030 - val_loss: 6.2414e-04
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0030 - val_loss: 5.8397e-04
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.0030 - val_loss: 5.6806e-04
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0028 - val_loss: 5.9807e-04
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0026 - val_loss: 5.6708e-04
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0028 - val_loss: 6.2195e-04
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0028 - val_loss: 5.3351e-04
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0030 - val_loss: 5.3452e-04
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 6.3654e-04
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0026 - val_loss: 5.2280e-04
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0028 - val_loss: 6.7801e-04
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0028 - val_loss: 5.8363e-04
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0026 - val_loss: 5.0648e-04
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0025 - val_loss: 5.0062e-04
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0025 - val_loss: 5.9529e-04
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0028 - val_loss: 4.9170e-04
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0025 - val_loss: 5.3352e-04
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 502us/step - loss: 0.0024 - val_loss: 5.6878e-04
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0022 - val_loss: 4.8698e-04
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0026 - val_loss: 4.9274e-04
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0024 - val_loss: 4.7954e-04
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0023 - val_loss: 5.0745e-04
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0023 - val_loss: 5.5049e-04
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0023 - val_loss: 5.4514e-04
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0024 - val_loss: 5.0166e-04
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0021 - val_loss: 4.9721e-04
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0021 - val_loss: 5.1595e-04
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0020 - val_loss: 4.7462e-04
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0022 - val_loss: 5.0000e-04
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0023 - val_loss: 4.8568e-04
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0024 - val_loss: 5.1235e-04
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0021 - val_loss: 4.7088e-04
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0021 - val_loss: 5.0335e-04
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0020 - val_loss: 5.3903e-04
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0019 - val_loss: 5.0135e-04
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0020 - val_loss: 5.5073e-04
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0019 - val_loss: 5.0803e-04
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0019 - val_loss: 4.9933e-04
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0020 - val_loss: 6.0936e-04
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0018 - val_loss: 5.3734e-04
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0021 - val_loss: 4.8737e-04
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0019 - val_loss: 4.8322e-04
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0018 - val_loss: 6.2390e-04
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0018 - val_loss: 5.5371e-04
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0018 - val_loss: 5.9039e-04
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0019 - val_loss: 4.9007e-04
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0020 - val_loss: 4.7367e-04
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0019 - val_loss: 4.6579e-04
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0018 - val_loss: 5.2859e-04
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0017 - val_loss: 4.5807e-04
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0017 - val_loss: 5.2901e-04
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0017 - val_loss: 4.8632e-04
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0018 - val_loss: 4.2463e-04
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0017 - val_loss: 4.5837e-04
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0017 - val_loss: 5.0185e-04
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0017 - val_loss: 6.1245e-04
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0016 - val_loss: 6.0365e-04
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0019 - val_loss: 4.6435e-04
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0016 - val_loss: 4.9160e-04
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0015 - val_loss: 5.3334e-04
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0018 - val_loss: 4.8659e-04
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0016 - val_loss: 4.7751e-04
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0015 - val_loss: 4.5946e-04
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0017 - val_loss: 5.6020e-04
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0014 - val_loss: 4.8858e-04
    Epoch 103/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0017 - val_loss: 4.9111e-04
    Epoch 104/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0016 - val_loss: 4.7014e-04
    Epoch 105/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0016 - val_loss: 4.3488e-04
    Epoch 106/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0016 - val_loss: 4.5564e-04
    Epoch 107/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0015 - val_loss: 4.7953e-04
    Epoch 108/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0015 - val_loss: 5.4949e-04
    Epoch 109/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0016 - val_loss: 4.9956e-04
    Epoch 110/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0015 - val_loss: 4.3580e-04
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 335us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4616 - val_loss: 0.0172
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0808 - val_loss: 0.0078
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 538us/step - loss: 0.0380 - val_loss: 0.0062
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 538us/step - loss: 0.0250 - val_loss: 0.0054
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0183 - val_loss: 0.0051
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0148 - val_loss: 0.0048
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0129 - val_loss: 0.0044
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0103 - val_loss: 0.0041
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0097 - val_loss: 0.0043
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0089 - val_loss: 0.0038
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0078 - val_loss: 0.0037
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0073 - val_loss: 0.0036
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0073 - val_loss: 0.0032
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0071 - val_loss: 0.0032
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0067 - val_loss: 0.0031
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0061 - val_loss: 0.0029
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0061 - val_loss: 0.0028
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0057 - val_loss: 0.0026
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0056 - val_loss: 0.0028
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0057 - val_loss: 0.0027
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0051 - val_loss: 0.0025
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0052 - val_loss: 0.0024
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0049 - val_loss: 0.0024
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0049 - val_loss: 0.0022
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0047 - val_loss: 0.0022
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0045 - val_loss: 0.0023
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0044 - val_loss: 0.0022
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0043 - val_loss: 0.0020
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0040 - val_loss: 0.0020
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0042 - val_loss: 0.0020
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0041 - val_loss: 0.0019
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0040 - val_loss: 0.0020
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0042 - val_loss: 0.0018
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0041 - val_loss: 0.0018
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0041 - val_loss: 0.0019
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0037 - val_loss: 0.0019
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0042 - val_loss: 0.0017
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0037 - val_loss: 0.0019
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0038 - val_loss: 0.0018
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0036 - val_loss: 0.0018
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0034 - val_loss: 0.0016
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0036 - val_loss: 0.0017
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0037 - val_loss: 0.0015
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0033 - val_loss: 0.0015
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0037 - val_loss: 0.0015
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0034 - val_loss: 0.0015
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0034 - val_loss: 0.0015
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0034 - val_loss: 0.0014
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0033 - val_loss: 0.0014
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0032 - val_loss: 0.0015
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0032 - val_loss: 0.0015
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0030 - val_loss: 0.0014
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0030 - val_loss: 0.0014
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0030 - val_loss: 0.0017
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0032 - val_loss: 0.0014
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0029 - val_loss: 0.0014
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0031 - val_loss: 0.0016
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0029 - val_loss: 0.0014
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 0.0015
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0026 - val_loss: 0.0012
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 0.0014
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0027 - val_loss: 0.0014
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0027 - val_loss: 0.0016
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0031 - val_loss: 0.0013
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0027 - val_loss: 0.0013
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0025 - val_loss: 0.0013
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0025 - val_loss: 0.0013
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0026 - val_loss: 0.0013
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0026 - val_loss: 0.0013
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0024 - val_loss: 0.0013
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0027 - val_loss: 0.0014
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0026 - val_loss: 0.0013
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0026 - val_loss: 0.0013
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0024 - val_loss: 0.0014
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0026 - val_loss: 0.0013
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 338us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 228us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4650 - val_loss: 0.0148
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0836 - val_loss: 0.0076
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0366 - val_loss: 0.0055
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0273 - val_loss: 0.0040
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0206 - val_loss: 0.0032
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0149 - val_loss: 0.0030
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0125 - val_loss: 0.0027
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0115 - val_loss: 0.0024
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0098 - val_loss: 0.0023
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0105 - val_loss: 0.0019
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0092 - val_loss: 0.0020
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0080 - val_loss: 0.0017
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0082 - val_loss: 0.0016
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0073 - val_loss: 0.0015
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0070 - val_loss: 0.0015
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0073 - val_loss: 0.0013
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0066 - val_loss: 0.0012
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0066 - val_loss: 0.0012
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0058 - val_loss: 0.0012
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0066 - val_loss: 0.0011
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0060 - val_loss: 0.0011
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 504us/step - loss: 0.0058 - val_loss: 0.0011
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0057 - val_loss: 0.0011
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0049 - val_loss: 0.0010
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0053 - val_loss: 9.5979e-04
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0053 - val_loss: 0.0011
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0053 - val_loss: 9.6288e-04
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0053 - val_loss: 9.3919e-04
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0049 - val_loss: 9.0662e-04
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0046 - val_loss: 9.0511e-04
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0051 - val_loss: 9.2219e-04
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0044 - val_loss: 8.7756e-04
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0044 - val_loss: 8.2298e-04
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 552us/step - loss: 0.0040 - val_loss: 7.9813e-04
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 537us/step - loss: 0.0045 - val_loss: 7.5959e-04
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0039 - val_loss: 7.9519e-04
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0042 - val_loss: 8.1922e-04
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0042 - val_loss: 8.0352e-04
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0043 - val_loss: 8.2510e-04
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0047 - val_loss: 7.5181e-04
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0039 - val_loss: 7.3766e-04
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0040 - val_loss: 7.6685e-04
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0040 - val_loss: 6.8456e-04
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0039 - val_loss: 7.2469e-04
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0042 - val_loss: 7.2900e-04
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0039 - val_loss: 6.7610e-04
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0036 - val_loss: 7.4520e-04
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0037 - val_loss: 6.4840e-04
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0036 - val_loss: 6.2857e-04
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0036 - val_loss: 6.4901e-04
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0034 - val_loss: 6.8107e-04
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0034 - val_loss: 6.9607e-04
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0033 - val_loss: 8.2347e-04
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0037 - val_loss: 6.5715e-04
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0036 - val_loss: 6.4929e-04
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0034 - val_loss: 6.6707e-04
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0035 - val_loss: 6.8389e-04
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0031 - val_loss: 7.0094e-04
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0032 - val_loss: 6.0296e-04
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0030 - val_loss: 6.1879e-04
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0032 - val_loss: 6.9654e-04
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0036 - val_loss: 6.5347e-04
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0031 - val_loss: 6.0014e-04
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0031 - val_loss: 7.2264e-04
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0031 - val_loss: 6.7448e-04
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 505us/step - loss: 0.0031 - val_loss: 6.2649e-04
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0031 - val_loss: 6.8709e-04
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0032 - val_loss: 6.4306e-04
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 7.0721e-04
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0028 - val_loss: 6.9180e-04
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0029 - val_loss: 6.6265e-04
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0029 - val_loss: 6.9076e-04
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 537us/step - loss: 0.0029 - val_loss: 7.0894e-04
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0029 - val_loss: 6.6444e-04
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0028 - val_loss: 6.6364e-04
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0028 - val_loss: 6.2871e-04
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0025 - val_loss: 6.3036e-04
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0030 - val_loss: 6.3796e-04
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0026 - val_loss: 6.2013e-04
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0026 - val_loss: 6.3910e-04
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0026 - val_loss: 5.8530e-04
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 492us/step - loss: 0.0030 - val_loss: 5.8419e-04
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0024 - val_loss: 5.9718e-04
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0027 - val_loss: 6.4777e-04
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0027 - val_loss: 5.9745e-04
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0024 - val_loss: 6.6335e-04
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0024 - val_loss: 6.8629e-04
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0022 - val_loss: 6.4242e-04
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0023 - val_loss: 6.4208e-04
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0024 - val_loss: 6.9601e-04
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0024 - val_loss: 5.9102e-04
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0025 - val_loss: 6.2734e-04
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0024 - val_loss: 6.7650e-04
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0027 - val_loss: 6.1623e-04
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 7.0282e-04
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0023 - val_loss: 7.2109e-04
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0023 - val_loss: 6.3272e-04
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0023 - val_loss: 8.1673e-04
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0022 - val_loss: 6.1501e-04
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 504us/step - loss: 0.0021 - val_loss: 6.6344e-04
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0023 - val_loss: 6.6701e-04
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0022 - val_loss: 6.2405e-04
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 344us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 224us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 1.0479 - val_loss: 0.0312
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.1567 - val_loss: 0.0144
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0775 - val_loss: 0.0104
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0546 - val_loss: 0.0082
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0384 - val_loss: 0.0074
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0293 - val_loss: 0.0068
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0219 - val_loss: 0.0060
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0190 - val_loss: 0.0053
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0168 - val_loss: 0.0050
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0156 - val_loss: 0.0046
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0150 - val_loss: 0.0044
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0122 - val_loss: 0.0040
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0116 - val_loss: 0.0038
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0118 - val_loss: 0.0034
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0095 - val_loss: 0.0034
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0109 - val_loss: 0.0033
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0093 - val_loss: 0.0030
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0090 - val_loss: 0.0030
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0090 - val_loss: 0.0029
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0090 - val_loss: 0.0026
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0078 - val_loss: 0.0027
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0077 - val_loss: 0.0026
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0078 - val_loss: 0.0025
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0074 - val_loss: 0.0025
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0077 - val_loss: 0.0025
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0073 - val_loss: 0.0022
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0074 - val_loss: 0.0023
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0071 - val_loss: 0.0023
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0069 - val_loss: 0.0023
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0063 - val_loss: 0.0021
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0063 - val_loss: 0.0021
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0069 - val_loss: 0.0022
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0060 - val_loss: 0.0020
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0064 - val_loss: 0.0019
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0058 - val_loss: 0.0022
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0063 - val_loss: 0.0020
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0059 - val_loss: 0.0021
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0059 - val_loss: 0.0020
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0059 - val_loss: 0.0021
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0054 - val_loss: 0.0018
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0056 - val_loss: 0.0020
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0054 - val_loss: 0.0017
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0055 - val_loss: 0.0019
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0061 - val_loss: 0.0018
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0055 - val_loss: 0.0018
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0053 - val_loss: 0.0018
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0053 - val_loss: 0.0017
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0055 - val_loss: 0.0016
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0054 - val_loss: 0.0017
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0051 - val_loss: 0.0017
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0051 - val_loss: 0.0018
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0048 - val_loss: 0.0018
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0050 - val_loss: 0.0016
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0051 - val_loss: 0.0018
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0046 - val_loss: 0.0017
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0052 - val_loss: 0.0018
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0051 - val_loss: 0.0016
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0043 - val_loss: 0.0017
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0047 - val_loss: 0.0016
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0044 - val_loss: 0.0017
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0047 - val_loss: 0.0018
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0045 - val_loss: 0.0017
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0046 - val_loss: 0.0017
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0042 - val_loss: 0.0016
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0045 - val_loss: 0.0016
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0040 - val_loss: 0.0016
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0045 - val_loss: 0.0015
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0041 - val_loss: 0.0016
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0039 - val_loss: 0.0016
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0045 - val_loss: 0.0016
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0045 - val_loss: 0.0017
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0041 - val_loss: 0.0016
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0041 - val_loss: 0.0016
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0039 - val_loss: 0.0016
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0045 - val_loss: 0.0015
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0041 - val_loss: 0.0016
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0044 - val_loss: 0.0016
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0038 - val_loss: 0.0016
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0039 - val_loss: 0.0016
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0036 - val_loss: 0.0017
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0039 - val_loss: 0.0016
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0038 - val_loss: 0.0015
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 553us/step - loss: 0.0036 - val_loss: 0.0016
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 508us/step - loss: 0.0041 - val_loss: 0.0015
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0035 - val_loss: 0.0016
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0037 - val_loss: 0.0016
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0037 - val_loss: 0.0016
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0037 - val_loss: 0.0018
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0042 - val_loss: 0.0015
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0038 - val_loss: 0.0016
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0034 - val_loss: 0.0017
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0037 - val_loss: 0.0016
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0037 - val_loss: 0.0016
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0037 - val_loss: 0.0016
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0038 - val_loss: 0.0016
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0035 - val_loss: 0.0016
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0034 - val_loss: 0.0016
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0032 - val_loss: 0.0016
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0035 - val_loss: 0.0016
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0032 - val_loss: 0.0016
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0034 - val_loss: 0.0016
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0033 - val_loss: 0.0016
    Epoch 103/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 539us/step - loss: 0.0031 - val_loss: 0.0016
    Epoch 104/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0034 - val_loss: 0.0016
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 342us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 224us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.5095 - val_loss: 0.0192
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0986 - val_loss: 0.0119
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0457 - val_loss: 0.0096
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0282 - val_loss: 0.0086
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0180 - val_loss: 0.0076
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0137 - val_loss: 0.0071
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0140 - val_loss: 0.0069
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0103 - val_loss: 0.0064
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0106 - val_loss: 0.0062
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0089 - val_loss: 0.0056
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0086 - val_loss: 0.0053
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0075 - val_loss: 0.0053
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0070 - val_loss: 0.0048
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0072 - val_loss: 0.0047
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0067 - val_loss: 0.0048
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0065 - val_loss: 0.0045
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0068 - val_loss: 0.0041
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0055 - val_loss: 0.0039
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0053 - val_loss: 0.0039
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0050 - val_loss: 0.0037
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0050 - val_loss: 0.0035
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0049 - val_loss: 0.0034
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0047 - val_loss: 0.0033
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0045 - val_loss: 0.0032
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0043 - val_loss: 0.0034
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0046 - val_loss: 0.0030
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0044 - val_loss: 0.0032
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0041 - val_loss: 0.0031
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0043 - val_loss: 0.0031
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0041 - val_loss: 0.0032
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0042 - val_loss: 0.0028
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0038 - val_loss: 0.0029
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0041 - val_loss: 0.0026
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0036 - val_loss: 0.0027
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0037 - val_loss: 0.0029
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0037 - val_loss: 0.0029
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0038 - val_loss: 0.0028
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0036 - val_loss: 0.0028
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0033 - val_loss: 0.0025
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0037 - val_loss: 0.0026
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0033 - val_loss: 0.0025
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0033 - val_loss: 0.0024
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0032 - val_loss: 0.0027
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0034 - val_loss: 0.0024
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0033 - val_loss: 0.0026
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0031 - val_loss: 0.0021
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0030 - val_loss: 0.0021
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0031 - val_loss: 0.0022
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0031 - val_loss: 0.0023
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0030 - val_loss: 0.0022
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0029 - val_loss: 0.0025
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0028 - val_loss: 0.0022
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0028 - val_loss: 0.0020
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0027 - val_loss: 0.0021
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0027 - val_loss: 0.0018
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0027 - val_loss: 0.0020
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0027 - val_loss: 0.0019
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0028 - val_loss: 0.0021
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0026 - val_loss: 0.0019
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0024 - val_loss: 0.0017
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0025 - val_loss: 0.0018
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0026 - val_loss: 0.0018
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0024 - val_loss: 0.0019
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0025 - val_loss: 0.0020
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0025 - val_loss: 0.0021
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0026 - val_loss: 0.0017
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0025 - val_loss: 0.0016
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0027 - val_loss: 0.0019
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0025 - val_loss: 0.0016
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0025 - val_loss: 0.0019
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0023 - val_loss: 0.0017
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0023 - val_loss: 0.0017
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0024 - val_loss: 0.0018
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0022 - val_loss: 0.0015
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0023 - val_loss: 0.0014
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0022 - val_loss: 0.0018
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0023 - val_loss: 0.0016
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0021 - val_loss: 0.0017
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0023 - val_loss: 0.0016
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0023 - val_loss: 0.0015
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0020 - val_loss: 0.0015
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0022 - val_loss: 0.0016
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0021 - val_loss: 0.0020
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0020 - val_loss: 0.0015
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0022 - val_loss: 0.0015
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0021 - val_loss: 0.0016
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0020 - val_loss: 0.0014
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0021 - val_loss: 0.0015
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0020 - val_loss: 0.0014
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0020 - val_loss: 0.0015
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0021 - val_loss: 0.0016
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0018 - val_loss: 0.0015
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0018 - val_loss: 0.0014
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0018 - val_loss: 0.0016
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0020 - val_loss: 0.0015
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0017 - val_loss: 0.0014
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0019 - val_loss: 0.0012
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0019 - val_loss: 0.0015
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0019 - val_loss: 0.0014
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0018 - val_loss: 0.0020
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0019 - val_loss: 0.0014
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0016 - val_loss: 0.0014
    Epoch 103/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0017 - val_loss: 0.0014
    Epoch 104/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0018 - val_loss: 0.0014
    Epoch 105/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0019 - val_loss: 0.0019
    Epoch 106/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0017 - val_loss: 0.0013
    Epoch 107/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0017 - val_loss: 0.0015
    Epoch 108/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0017 - val_loss: 0.0015
    Epoch 109/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - loss: 0.0018 - val_loss: 0.0013
    Epoch 110/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 552us/step - loss: 0.0015 - val_loss: 0.0017
    Epoch 111/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0017 - val_loss: 0.0013
    Epoch 112/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 544us/step - loss: 0.0016 - val_loss: 0.0017
    Epoch 113/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0016 - val_loss: 0.0016
    Epoch 114/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0017 - val_loss: 0.0013
    Epoch 115/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0017 - val_loss: 0.0013
    Epoch 116/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0016 - val_loss: 0.0014
    Epoch 117/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0016 - val_loss: 0.0014
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 333us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 233us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4986 - val_loss: 0.0220
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0778 - val_loss: 0.0123
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0411 - val_loss: 0.0095
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0267 - val_loss: 0.0084
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0210 - val_loss: 0.0077
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 507us/step - loss: 0.0148 - val_loss: 0.0072
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0149 - val_loss: 0.0070
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0131 - val_loss: 0.0065
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0110 - val_loss: 0.0065
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 600us/step - loss: 0.0115 - val_loss: 0.0061
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 540us/step - loss: 0.0114 - val_loss: 0.0060
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0095 - val_loss: 0.0058
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0101 - val_loss: 0.0060
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0086 - val_loss: 0.0055
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0081 - val_loss: 0.0053
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0078 - val_loss: 0.0052
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0081 - val_loss: 0.0049
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0079 - val_loss: 0.0050
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0071 - val_loss: 0.0047
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0072 - val_loss: 0.0047
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0067 - val_loss: 0.0042
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0067 - val_loss: 0.0044
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0064 - val_loss: 0.0041
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0065 - val_loss: 0.0042
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0063 - val_loss: 0.0041
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0063 - val_loss: 0.0040
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0059 - val_loss: 0.0037
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0056 - val_loss: 0.0038
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0056 - val_loss: 0.0037
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0054 - val_loss: 0.0037
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0056 - val_loss: 0.0035
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0053 - val_loss: 0.0036
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 509us/step - loss: 0.0051 - val_loss: 0.0034
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0054 - val_loss: 0.0034
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 0.0051 - val_loss: 0.0037
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0047 - val_loss: 0.0033
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0053 - val_loss: 0.0033
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0049 - val_loss: 0.0033
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0049 - val_loss: 0.0032
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0048 - val_loss: 0.0033
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0045 - val_loss: 0.0033
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0043 - val_loss: 0.0029
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0049 - val_loss: 0.0031
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0045 - val_loss: 0.0032
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0046 - val_loss: 0.0032
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0046 - val_loss: 0.0033
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 507us/step - loss: 0.0046 - val_loss: 0.0030
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0042 - val_loss: 0.0032
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 0.0044 - val_loss: 0.0033
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0042 - val_loss: 0.0030
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0041 - val_loss: 0.0028
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0042 - val_loss: 0.0030
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0040 - val_loss: 0.0030
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0040 - val_loss: 0.0028
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0037 - val_loss: 0.0031
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0043 - val_loss: 0.0031
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0038 - val_loss: 0.0031
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0038 - val_loss: 0.0030
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0038 - val_loss: 0.0029
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0038 - val_loss: 0.0029
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0039 - val_loss: 0.0028
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0040 - val_loss: 0.0029
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0039 - val_loss: 0.0031
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0036 - val_loss: 0.0030
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 495us/step - loss: 0.0037 - val_loss: 0.0028
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0034 - val_loss: 0.0032
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0034 - val_loss: 0.0033
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0035 - val_loss: 0.0031
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0034 - val_loss: 0.0028
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0033 - val_loss: 0.0029
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0033 - val_loss: 0.0029
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0034 - val_loss: 0.0029
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 0.0034 - val_loss: 0.0027
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0035 - val_loss: 0.0029
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0036 - val_loss: 0.0030
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0033 - val_loss: 0.0030
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0034 - val_loss: 0.0028
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0036 - val_loss: 0.0029
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0034 - val_loss: 0.0031
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0033 - val_loss: 0.0028
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0033 - val_loss: 0.0029
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0031 - val_loss: 0.0028
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0032 - val_loss: 0.0029
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0030 - val_loss: 0.0030
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0031 - val_loss: 0.0029
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0031 - val_loss: 0.0029
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0031 - val_loss: 0.0030
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0030 - val_loss: 0.0028
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0031 - val_loss: 0.0028
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0028 - val_loss: 0.0030
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0030 - val_loss: 0.0031
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0032 - val_loss: 0.0029
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0028 - val_loss: 0.0029
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 350us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 229us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.5598 - val_loss: 0.0157
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 545us/step - loss: 0.1071 - val_loss: 0.0058
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 540us/step - loss: 0.0500 - val_loss: 0.0030
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 544us/step - loss: 0.0338 - val_loss: 0.0018
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 543us/step - loss: 0.0163 - val_loss: 0.0013
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0106 - val_loss: 9.8512e-04
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 543us/step - loss: 0.0090 - val_loss: 8.2897e-04
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0065 - val_loss: 7.8668e-04
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 543us/step - loss: 0.0053 - val_loss: 7.6147e-04
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 540us/step - loss: 0.0046 - val_loss: 7.3455e-04
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 539us/step - loss: 0.0036 - val_loss: 6.9298e-04
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0037 - val_loss: 6.7329e-04
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0036 - val_loss: 6.8181e-04
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0029 - val_loss: 6.5268e-04
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0026 - val_loss: 6.3365e-04
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0023 - val_loss: 6.2997e-04
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0025 - val_loss: 6.0665e-04
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0021 - val_loss: 5.9735e-04
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0022 - val_loss: 5.8282e-04
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0019 - val_loss: 5.7123e-04
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0018 - val_loss: 5.2732e-04
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0021 - val_loss: 5.4949e-04
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0019 - val_loss: 5.5645e-04
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0018 - val_loss: 5.3367e-04
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0018 - val_loss: 4.7790e-04
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0016 - val_loss: 5.3853e-04
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0016 - val_loss: 5.0247e-04
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0016 - val_loss: 4.6440e-04
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0015 - val_loss: 4.5584e-04
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0014 - val_loss: 4.6568e-04
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0015 - val_loss: 4.8706e-04
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0015 - val_loss: 4.6016e-04
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0014 - val_loss: 4.7107e-04
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0014 - val_loss: 4.3498e-04
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0014 - val_loss: 4.5103e-04
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0014 - val_loss: 4.2764e-04
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0012 - val_loss: 4.5204e-04
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0013 - val_loss: 4.4789e-04
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0012 - val_loss: 4.0832e-04
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0012 - val_loss: 4.5429e-04
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0013 - val_loss: 4.4293e-04
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0012 - val_loss: 4.1620e-04
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0012 - val_loss: 4.2252e-04
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0012 - val_loss: 3.9411e-04
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0012 - val_loss: 3.9466e-04
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0011 - val_loss: 3.8754e-04
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0011 - val_loss: 3.7984e-04
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0012 - val_loss: 3.6600e-04
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0011 - val_loss: 3.6320e-04
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0011 - val_loss: 3.6891e-04
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0011 - val_loss: 3.4482e-04
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 9.9633e-04 - val_loss: 3.4921e-04
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0010 - val_loss: 3.4913e-04
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0010 - val_loss: 3.6842e-04
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 9.4881e-04 - val_loss: 3.4932e-04
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0010 - val_loss: 3.3458e-04
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 9.4144e-04 - val_loss: 3.2061e-04
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0010 - val_loss: 4.1205e-04
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0010 - val_loss: 3.4300e-04
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 9.9403e-04 - val_loss: 3.3583e-04
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 9.6981e-04 - val_loss: 3.5615e-04
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 9.7370e-04 - val_loss: 3.4013e-04
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 8.6865e-04 - val_loss: 3.5070e-04
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 8.5598e-04 - val_loss: 3.1207e-04
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 9.1074e-04 - val_loss: 4.6154e-04
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 8.8714e-04 - val_loss: 2.9860e-04
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 8.7175e-04 - val_loss: 3.0632e-04
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 9.2629e-04 - val_loss: 3.3377e-04
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 8.7798e-04 - val_loss: 3.2717e-04
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 7.9528e-04 - val_loss: 3.0341e-04
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 8.8259e-04 - val_loss: 2.9672e-04
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 8.7207e-04 - val_loss: 3.4279e-04
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 8.0036e-04 - val_loss: 3.1302e-04
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 8.5986e-04 - val_loss: 2.9655e-04
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 8.5940e-04 - val_loss: 3.2665e-04
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 8.0988e-04 - val_loss: 2.9423e-04
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 7.3519e-04 - val_loss: 3.0448e-04
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 7.5155e-04 - val_loss: 2.9300e-04
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 8.0842e-04 - val_loss: 2.9964e-04
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 7.5369e-04 - val_loss: 3.5232e-04
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 8.1116e-04 - val_loss: 2.8199e-04
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 7.3744e-04 - val_loss: 2.9262e-04
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 507us/step - loss: 7.8302e-04 - val_loss: 2.9370e-04
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 8.6557e-04 - val_loss: 2.9401e-04
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 7.4431e-04 - val_loss: 2.8175e-04
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 8.1440e-04 - val_loss: 3.0261e-04
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 7.6763e-04 - val_loss: 2.8681e-04
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 7.9964e-04 - val_loss: 2.7859e-04
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 8.1623e-04 - val_loss: 2.9993e-04
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 6.7423e-04 - val_loss: 2.9741e-04
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 7.7879e-04 - val_loss: 2.8437e-04
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 7.7511e-04 - val_loss: 3.1725e-04
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 8.0305e-04 - val_loss: 3.1124e-04
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 7.2067e-04 - val_loss: 3.1853e-04
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 7.6761e-04 - val_loss: 2.8007e-04
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 6.9813e-04 - val_loss: 2.8523e-04
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 8.1591e-04 - val_loss: 3.2905e-04
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 7.6752e-04 - val_loss: 2.7487e-04
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 6.6052e-04 - val_loss: 3.0664e-04
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 7.8494e-04 - val_loss: 2.8954e-04
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 7.5902e-04 - val_loss: 2.8118e-04
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 7.4071e-04 - val_loss: 3.1599e-04
    Epoch 103/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 6.7289e-04 - val_loss: 3.4667e-04
    Epoch 104/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 6.5966e-04 - val_loss: 3.0563e-04
    Epoch 105/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 6.8473e-04 - val_loss: 3.1026e-04
    Epoch 106/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 7.0518e-04 - val_loss: 2.9491e-04
    Epoch 107/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 6.8591e-04 - val_loss: 2.8822e-04
    Epoch 108/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 6.0082e-04 - val_loss: 2.8070e-04
    Epoch 109/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 6.1940e-04 - val_loss: 2.7343e-04
    Epoch 110/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 6.6513e-04 - val_loss: 2.8766e-04
    Epoch 111/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 7.2770e-04 - val_loss: 2.8387e-04
    Epoch 112/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 6.7501e-04 - val_loss: 2.8332e-04
    Epoch 113/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 6.1906e-04 - val_loss: 2.9328e-04
    Epoch 114/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 6.6476e-04 - val_loss: 2.8464e-04
    Epoch 115/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 6.7171e-04 - val_loss: 2.7843e-04
    Epoch 116/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 6.8297e-04 - val_loss: 2.7629e-04
    Epoch 117/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 6.7914e-04 - val_loss: 3.0022e-04
    Epoch 118/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 6.5604e-04 - val_loss: 2.8570e-04
    Epoch 119/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 6.2802e-04 - val_loss: 2.8226e-04
    Epoch 120/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 6.2647e-04 - val_loss: 2.7962e-04
    Epoch 121/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 6.0283e-04 - val_loss: 2.9418e-04
    Epoch 122/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 5.9896e-04 - val_loss: 2.8723e-04
    Epoch 123/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 5.8245e-04 - val_loss: 2.8864e-04
    Epoch 124/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 6.1013e-04 - val_loss: 2.7735e-04
    Epoch 125/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 6.3487e-04 - val_loss: 3.0922e-04
    Epoch 126/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 6.6973e-04 - val_loss: 3.0993e-04
    Epoch 127/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 5.5109e-04 - val_loss: 2.8917e-04
    Epoch 128/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 6.4745e-04 - val_loss: 2.8705e-04
    Epoch 129/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 6.2997e-04 - val_loss: 3.0397e-04
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 336us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.7258 - val_loss: 0.0198
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.1399 - val_loss: 0.0125
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0636 - val_loss: 0.0085
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0315 - val_loss: 0.0068
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0243 - val_loss: 0.0062
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0163 - val_loss: 0.0056
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0139 - val_loss: 0.0052
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0110 - val_loss: 0.0050
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0097 - val_loss: 0.0048
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0085 - val_loss: 0.0046
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0070 - val_loss: 0.0044
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0066 - val_loss: 0.0041
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0065 - val_loss: 0.0042
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0057 - val_loss: 0.0041
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0062 - val_loss: 0.0040
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0058 - val_loss: 0.0038
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0054 - val_loss: 0.0036
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0053 - val_loss: 0.0034
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0050 - val_loss: 0.0031
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0050 - val_loss: 0.0035
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0052 - val_loss: 0.0030
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0043 - val_loss: 0.0031
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0045 - val_loss: 0.0029
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0045 - val_loss: 0.0028
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0042 - val_loss: 0.0027
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0040 - val_loss: 0.0026
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0042 - val_loss: 0.0027
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0037 - val_loss: 0.0026
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0036 - val_loss: 0.0024
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0036 - val_loss: 0.0023
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0036 - val_loss: 0.0025
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0033 - val_loss: 0.0022
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0034 - val_loss: 0.0023
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0034 - val_loss: 0.0020
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0033 - val_loss: 0.0020
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0032 - val_loss: 0.0021
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0031 - val_loss: 0.0018
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0030 - val_loss: 0.0018
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0031 - val_loss: 0.0018
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0030 - val_loss: 0.0019
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0029 - val_loss: 0.0018
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0028 - val_loss: 0.0018
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0028 - val_loss: 0.0018
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0028 - val_loss: 0.0020
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0026 - val_loss: 0.0016
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0028 - val_loss: 0.0016
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0024 - val_loss: 0.0015
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0027 - val_loss: 0.0016
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0025 - val_loss: 0.0015
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0024 - val_loss: 0.0017
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0026 - val_loss: 0.0015
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0024 - val_loss: 0.0014
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0026 - val_loss: 0.0016
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0023 - val_loss: 0.0014
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0025 - val_loss: 0.0012
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0025 - val_loss: 0.0015
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0022 - val_loss: 0.0014
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0021 - val_loss: 0.0014
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0025 - val_loss: 0.0015
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0021 - val_loss: 0.0012
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0022 - val_loss: 0.0014
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0021 - val_loss: 0.0012
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0022 - val_loss: 0.0013
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0021 - val_loss: 0.0014
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0022 - val_loss: 0.0012
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0019 - val_loss: 0.0014
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0023 - val_loss: 0.0012
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0022 - val_loss: 0.0013
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0020 - val_loss: 0.0013
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0019 - val_loss: 0.0012
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0021 - val_loss: 0.0013
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0019 - val_loss: 0.0013
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0019 - val_loss: 0.0013
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0018 - val_loss: 0.0012
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0019 - val_loss: 0.0011
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0020 - val_loss: 0.0012
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0020 - val_loss: 0.0012
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0019 - val_loss: 0.0012
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0021 - val_loss: 0.0013
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0018 - val_loss: 0.0012
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0018 - val_loss: 0.0011
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0018 - val_loss: 0.0011
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0018 - val_loss: 0.0011
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0019 - val_loss: 0.0011
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0018 - val_loss: 0.0011
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0018 - val_loss: 0.0012
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0018 - val_loss: 0.0012
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0017 - val_loss: 0.0011
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0017 - val_loss: 0.0012
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0015 - val_loss: 0.0013
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0017 - val_loss: 0.0012
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0016 - val_loss: 0.0012
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0017 - val_loss: 0.0012
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0017 - val_loss: 0.0011
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0016 - val_loss: 0.0012
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0017 - val_loss: 0.0011
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0017 - val_loss: 0.0012
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0016 - val_loss: 0.0011
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0016 - val_loss: 0.0011
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0016 - val_loss: 0.0011
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0017 - val_loss: 0.0011
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 342us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 229us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.4163 - val_loss: 0.0232
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0943 - val_loss: 0.0104
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 542us/step - loss: 0.0415 - val_loss: 0.0071
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 601us/step - loss: 0.0297 - val_loss: 0.0064
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0220 - val_loss: 0.0055
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0160 - val_loss: 0.0053
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0148 - val_loss: 0.0049
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0138 - val_loss: 0.0047
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0124 - val_loss: 0.0045
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0107 - val_loss: 0.0040
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0110 - val_loss: 0.0038
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0099 - val_loss: 0.0034
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0093 - val_loss: 0.0035
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0087 - val_loss: 0.0031
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0096 - val_loss: 0.0032
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0080 - val_loss: 0.0028
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 537us/step - loss: 0.0082 - val_loss: 0.0030
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - loss: 0.0075 - val_loss: 0.0030
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - loss: 0.0075 - val_loss: 0.0026
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 536us/step - loss: 0.0071 - val_loss: 0.0026
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0073 - val_loss: 0.0025
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0069 - val_loss: 0.0028
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0067 - val_loss: 0.0027
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0067 - val_loss: 0.0026
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0059 - val_loss: 0.0025
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0058 - val_loss: 0.0026
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 539us/step - loss: 0.0060 - val_loss: 0.0024
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0063 - val_loss: 0.0023
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 538us/step - loss: 0.0065 - val_loss: 0.0022
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0059 - val_loss: 0.0023
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0052 - val_loss: 0.0020
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0058 - val_loss: 0.0022
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0053 - val_loss: 0.0022
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.0053 - val_loss: 0.0021
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0052 - val_loss: 0.0020
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0052 - val_loss: 0.0019
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0053 - val_loss: 0.0020
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0047 - val_loss: 0.0019
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0050 - val_loss: 0.0020
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0047 - val_loss: 0.0018
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 534us/step - loss: 0.0052 - val_loss: 0.0020
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0050 - val_loss: 0.0019
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0051 - val_loss: 0.0019
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0049 - val_loss: 0.0019
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0050 - val_loss: 0.0021
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0052 - val_loss: 0.0018
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0045 - val_loss: 0.0020
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0045 - val_loss: 0.0020
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0053 - val_loss: 0.0018
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0041 - val_loss: 0.0019
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0044 - val_loss: 0.0017
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0042 - val_loss: 0.0018
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0043 - val_loss: 0.0018
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0043 - val_loss: 0.0020
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0045 - val_loss: 0.0019
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0046 - val_loss: 0.0018
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0044 - val_loss: 0.0017
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0043 - val_loss: 0.0016
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0044 - val_loss: 0.0018
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0042 - val_loss: 0.0018
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0044 - val_loss: 0.0019
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0040 - val_loss: 0.0018
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0039 - val_loss: 0.0019
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0040 - val_loss: 0.0016
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0040 - val_loss: 0.0016
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 69/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0036 - val_loss: 0.0016
    Epoch 70/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 508us/step - loss: 0.0041 - val_loss: 0.0017
    Epoch 71/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0038 - val_loss: 0.0017
    Epoch 72/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0042 - val_loss: 0.0017
    Epoch 73/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0039 - val_loss: 0.0020
    Epoch 74/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 75/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 76/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 0.0037 - val_loss: 0.0017
    Epoch 77/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0033 - val_loss: 0.0018
    Epoch 78/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0041 - val_loss: 0.0019
    Epoch 79/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0038 - val_loss: 0.0016
    Epoch 80/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0037 - val_loss: 0.0017
    Epoch 81/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 512us/step - loss: 0.0035 - val_loss: 0.0016
    Epoch 82/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0034 - val_loss: 0.0016
    Epoch 83/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0038 - val_loss: 0.0016
    Epoch 84/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0035 - val_loss: 0.0018
    Epoch 85/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 509us/step - loss: 0.0033 - val_loss: 0.0016
    Epoch 86/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 507us/step - loss: 0.0035 - val_loss: 0.0018
    Epoch 87/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0032 - val_loss: 0.0016
    Epoch 88/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0036 - val_loss: 0.0016
    Epoch 89/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0033 - val_loss: 0.0018
    Epoch 90/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0038 - val_loss: 0.0019
    Epoch 91/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 0.0040 - val_loss: 0.0017
    Epoch 92/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0033 - val_loss: 0.0015
    Epoch 93/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 505us/step - loss: 0.0034 - val_loss: 0.0018
    Epoch 94/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0031 - val_loss: 0.0017
    Epoch 95/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0031 - val_loss: 0.0015
    Epoch 96/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0032 - val_loss: 0.0018
    Epoch 97/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 508us/step - loss: 0.0034 - val_loss: 0.0019
    Epoch 98/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0032 - val_loss: 0.0017
    Epoch 99/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0030 - val_loss: 0.0017
    Epoch 100/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0029 - val_loss: 0.0018
    Epoch 101/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0031 - val_loss: 0.0018
    Epoch 102/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0030 - val_loss: 0.0018
    Epoch 103/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 516us/step - loss: 0.0033 - val_loss: 0.0018
    Epoch 104/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0028 - val_loss: 0.0016
    Epoch 105/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0029 - val_loss: 0.0016
    Epoch 106/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0033 - val_loss: 0.0017
    Epoch 107/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0029 - val_loss: 0.0016
    Epoch 108/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0029 - val_loss: 0.0015
    Epoch 109/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0033 - val_loss: 0.0016
    Epoch 110/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 513us/step - loss: 0.0033 - val_loss: 0.0017
    Epoch 111/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0029 - val_loss: 0.0018
    Epoch 112/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 511us/step - loss: 0.0029 - val_loss: 0.0017
    Epoch 113/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0027 - val_loss: 0.0018
    Epoch 114/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0030 - val_loss: 0.0018
    Epoch 115/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 510us/step - loss: 0.0027 - val_loss: 0.0018
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 333us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 230us/step
    Epoch 1/200


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - loss: 0.5705 - val_loss: 0.0384
    Epoch 2/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 535us/step - loss: 0.1213 - val_loss: 0.0276
    Epoch 3/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 537us/step - loss: 0.0679 - val_loss: 0.0268
    Epoch 4/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0468 - val_loss: 0.0261
    Epoch 5/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0429 - val_loss: 0.0240
    Epoch 6/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0369 - val_loss: 0.0224
    Epoch 7/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0309 - val_loss: 0.0216
    Epoch 8/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0278 - val_loss: 0.0207
    Epoch 9/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0263 - val_loss: 0.0199
    Epoch 10/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0278 - val_loss: 0.0189
    Epoch 11/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0250 - val_loss: 0.0184
    Epoch 12/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0207 - val_loss: 0.0174
    Epoch 13/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0224 - val_loss: 0.0173
    Epoch 14/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0215 - val_loss: 0.0178
    Epoch 15/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0211 - val_loss: 0.0170
    Epoch 16/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0199 - val_loss: 0.0169
    Epoch 17/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0184 - val_loss: 0.0169
    Epoch 18/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0179 - val_loss: 0.0161
    Epoch 19/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0195 - val_loss: 0.0152
    Epoch 20/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0201 - val_loss: 0.0164
    Epoch 21/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0175 - val_loss: 0.0161
    Epoch 22/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0196 - val_loss: 0.0167
    Epoch 23/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0162 - val_loss: 0.0156
    Epoch 24/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0147 - val_loss: 0.0153
    Epoch 25/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0144 - val_loss: 0.0155
    Epoch 26/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 519us/step - loss: 0.0167 - val_loss: 0.0150
    Epoch 27/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0158 - val_loss: 0.0149
    Epoch 28/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0144 - val_loss: 0.0141
    Epoch 29/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0140 - val_loss: 0.0149
    Epoch 30/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0160 - val_loss: 0.0143
    Epoch 31/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0166 - val_loss: 0.0146
    Epoch 32/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0145 - val_loss: 0.0147
    Epoch 33/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 528us/step - loss: 0.0133 - val_loss: 0.0149
    Epoch 34/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 515us/step - loss: 0.0133 - val_loss: 0.0148
    Epoch 35/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0139 - val_loss: 0.0146
    Epoch 36/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0126 - val_loss: 0.0146
    Epoch 37/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 533us/step - loss: 0.0132 - val_loss: 0.0140
    Epoch 38/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0141 - val_loss: 0.0141
    Epoch 39/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0125 - val_loss: 0.0141
    Epoch 40/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0132 - val_loss: 0.0143
    Epoch 41/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0126 - val_loss: 0.0141
    Epoch 42/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0125 - val_loss: 0.0137
    Epoch 43/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 520us/step - loss: 0.0132 - val_loss: 0.0131
    Epoch 44/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 529us/step - loss: 0.0121 - val_loss: 0.0138
    Epoch 45/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 522us/step - loss: 0.0121 - val_loss: 0.0136
    Epoch 46/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0113 - val_loss: 0.0145
    Epoch 47/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0118 - val_loss: 0.0143
    Epoch 48/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 531us/step - loss: 0.0136 - val_loss: 0.0125
    Epoch 49/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 517us/step - loss: 0.0115 - val_loss: 0.0130
    Epoch 50/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0121 - val_loss: 0.0143
    Epoch 51/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0117 - val_loss: 0.0136
    Epoch 52/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0119 - val_loss: 0.0134
    Epoch 53/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0107 - val_loss: 0.0132
    Epoch 54/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0116 - val_loss: 0.0137
    Epoch 55/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 530us/step - loss: 0.0122 - val_loss: 0.0137
    Epoch 56/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 526us/step - loss: 0.0108 - val_loss: 0.0134
    Epoch 57/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 527us/step - loss: 0.0104 - val_loss: 0.0127
    Epoch 58/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 547us/step - loss: 0.0110 - val_loss: 0.0132
    Epoch 59/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 558us/step - loss: 0.0105 - val_loss: 0.0142
    Epoch 60/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 514us/step - loss: 0.0107 - val_loss: 0.0134
    Epoch 61/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0104 - val_loss: 0.0140
    Epoch 62/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 525us/step - loss: 0.0094 - val_loss: 0.0137
    Epoch 63/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0109 - val_loss: 0.0132
    Epoch 64/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 521us/step - loss: 0.0096 - val_loss: 0.0129
    Epoch 65/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 524us/step - loss: 0.0094 - val_loss: 0.0127
    Epoch 66/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 532us/step - loss: 0.0100 - val_loss: 0.0138
    Epoch 67/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 523us/step - loss: 0.0094 - val_loss: 0.0136
    Epoch 68/200
    [1m89/89[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 518us/step - loss: 0.0091 - val_loss: 0.0128
    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 342us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 233us/step





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
      <th>r2: Simple</th>
      <th>r2: Multi</th>
      <th>r2: Neural</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>0.80</td>
      <td>0.93</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>0.92</td>
      <td>0.96</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>UK</th>
      <td>0.64</td>
      <td>0.88</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>France</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>0.74</td>
      <td>0.88</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>0.77</td>
      <td>0.91</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>0.66</td>
      <td>0.85</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>0.17</td>
      <td>0.67</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>0.70</td>
      <td>0.87</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>0.58</td>
      <td>0.81</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>0.52</td>
      <td>0.80</td>
      <td>0.66</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.68</td>
      <td>0.87</td>
      <td>0.82</td>
    </tr>
  </tbody>
</table>
</div>




```python
def trader(target, t, std):
    data = predictor('multi', target, t)[0]
    name = target
    target = codes[target]
    data['spread'] = data['prediction'] - data[target]
    threshold = data['spread'].std() * std
    data['signal'] = 0
    data.loc[data[target] > data['prediction'] + threshold , 'signal'] = -1
    data.loc[data[target] < data['prediction'] - threshold, 'signal'] = 1
    data['return'] = data[target].diff(1)
    data['signal_return'] = data['return'] * data['signal']
    data['cum_return'] = data['signal_return'].cumsum()
    total_return = round(list(data['cum_return'])[-1], 2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(data.index, data['spread'], label='Spread')
    axs[0].axhspan(-threshold, threshold, color='lightblue', alpha=1)
    axs[0].set_ylabel('Prediction - Actual')
    axs[0].set_title(f'{name} Signal Generation')
    axs[1].plot(data.index, data['cum_return'], label='Cumulative Return', color='green')
    axs[1].set_ylabel('Cumulative Return')
    axs[1].set_title(f'{name} Trading Performance')
    date_format = DateFormatter('%Y-%m')
    axs[0].xaxis.set_major_formatter(date_format)
    axs[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.show()
    return total_return

trader('US', 100, .25)
```


    
![png](G10_RV_files/G10_RV_3_0.png)
    





    1.2




```python
def momentum(ticker, window, stds):
    name = ticker
    ticker = codes[ticker]
    data = df[[ticker]].copy()
    data['MA'] = data[ticker].rolling(window).mean()
    data['upper'] = data['MA'] + data[ticker].rolling(window).std() * stds
    data['lower'] = data['MA'] - data[ticker].rolling(window).std() * stds
    data = data.dropna()
    data['signal'] = 0
    data.loc[data[ticker] > data['upper'], 'signal'] = 1
    data.loc[data[ticker] < data['lower'], 'signal'] = -1
    data['signal'] = data['signal'].shift(1)
    data['return'] = data[ticker].diff(1)
    data['signal_return'] = data['return'] * data['signal']
    data['cum_return'] = data['signal_return'].cumsum()
    total_return = round(list(data['cum_return'])[-1], 2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(data.index[-250:], data[ticker].iloc[-250:], label=ticker, color='blue')
    axs[0].plot(data.index[-250:], data['MA'].iloc[-250:], label='MA', color='black')
    axs[0].plot(data.index[-250:], data['upper'].iloc[-250:], label='Upper Band', linestyle='--', color='red')
    axs[0].plot(data.index[-250:], data['lower'].iloc[-250:], label='Lower Band', linestyle='--', color='green')
    axs[0].set_title(f"{name} Signal Generation (Zoomed)")
    axs[0].legend()
    axs[1].plot(data.index, data['cum_return'], label='Trading Performance', color='green')
    axs[1].set_title(f'{name} Cumulative Return')
    date_format = DateFormatter('%Y-%m')
    axs[0].xaxis.set_major_formatter(date_format)
    axs[1].xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.show()
    return total_return

momentum('UK', 25, .5)
```


    
![png](G10_RV_files/G10_RV_4_0.png)
    





    0.05




```python
trading = pd.DataFrame()
trading['Country'] = codes.keys()
trading['Predictive'] = trading['Country'].apply(lambda x: trader(x, 100, .25))
trading['Momentum'] = trading['Country'].apply(lambda x: momentum(x, 25, .5))
trading = trading.set_index('Country')
trading.loc['Mean'] = round(trading.mean(), 2)
```


    
![png](G10_RV_files/G10_RV_5_0.png)
    



    
![png](G10_RV_files/G10_RV_5_1.png)
    



    
![png](G10_RV_files/G10_RV_5_2.png)
    



    
![png](G10_RV_files/G10_RV_5_3.png)
    



    
![png](G10_RV_files/G10_RV_5_4.png)
    



    
![png](G10_RV_files/G10_RV_5_5.png)
    



    
![png](G10_RV_files/G10_RV_5_6.png)
    



    
![png](G10_RV_files/G10_RV_5_7.png)
    



    
![png](G10_RV_files/G10_RV_5_8.png)
    



    
![png](G10_RV_files/G10_RV_5_9.png)
    



    
![png](G10_RV_files/G10_RV_5_10.png)
    



    
![png](G10_RV_files/G10_RV_5_11.png)
    



    
![png](G10_RV_files/G10_RV_5_12.png)
    



    
![png](G10_RV_files/G10_RV_5_13.png)
    



    
![png](G10_RV_files/G10_RV_5_14.png)
    



    
![png](G10_RV_files/G10_RV_5_15.png)
    



    
![png](G10_RV_files/G10_RV_5_16.png)
    



    
![png](G10_RV_files/G10_RV_5_17.png)
    



    
![png](G10_RV_files/G10_RV_5_18.png)
    



    
![png](G10_RV_files/G10_RV_5_19.png)
    



    
![png](G10_RV_files/G10_RV_5_20.png)
    



    
![png](G10_RV_files/G10_RV_5_21.png)
    



```python
trading
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
      <th>Predictive</th>
      <th>Momentum</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>US</th>
      <td>1.20</td>
      <td>2.13</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>-1.34</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>UK</th>
      <td>-2.65</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>France</th>
      <td>3.45</td>
      <td>4.19</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>0.98</td>
      <td>-0.57</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>0.34</td>
      <td>1.92</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>-0.93</td>
      <td>10.88</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>0.44</td>
      <td>-1.09</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>0.78</td>
      <td>4.70</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>-0.59</td>
      <td>8.55</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>-2.62</td>
      <td>6.15</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>-0.09</td>
      <td>3.40</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
