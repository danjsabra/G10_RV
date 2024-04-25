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
        model.fit(training_X, training_y, epochs=200, validation_split=0.2, callbacks=[early_stopping], verbose=0)
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

    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 341us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 236us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 354us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 229us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 417us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 247us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 366us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 231us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 341us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 444us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 365us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 412us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 248us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 236us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 364us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 238us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 344us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 235us/step


    /Users/jv/anaconda3/envs/neural_network/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)


    [1m111/111[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 355us/step
    [1m49/49[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 234us/step





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
      <td>0.90</td>
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
      <td>0.80</td>
    </tr>
    <tr>
      <th>France</th>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>0.74</td>
      <td>0.88</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>0.77</td>
      <td>0.91</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>0.66</td>
      <td>0.85</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>0.17</td>
      <td>0.67</td>
      <td>0.61</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>0.70</td>
      <td>0.87</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>0.58</td>
      <td>0.81</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>0.52</td>
      <td>0.80</td>
      <td>0.72</td>
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
