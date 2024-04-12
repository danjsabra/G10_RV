import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Dicts to identify which countries go with which tickers (all are 10-year Govt yields)
country_to_ticker = {}
country_to_ticker['US'] = 'USGG10YR'
country_to_ticker['Germany'] = 'GDBR10'
country_to_ticker['UK'] = 'GUKG10'
country_to_ticker['France'] = 'GFRN10'
country_to_ticker['Australia'] = 'GACGB10'
country_to_ticker['Canada'] = 'GCAN10YR'
country_to_ticker['New Zealand'] = 'GNZGB10'
country_to_ticker['Japan'] = 'JGBS10'
country_to_ticker['Switzerland'] = 'GSWISS10'
country_to_ticker['Norway'] = 'GNOR10YR'
country_to_ticker['Italy'] = 'GBTPGR10'

ticker_to_country = {}
for country, ticker in country_to_ticker.items():
    ticker_to_country[ticker] = country

# Assuming you have an Excel file named 'data.xlsx'
excel_file = pd.ExcelFile('G10_RV.xlsx')

# Get the first 10 sheet names
sheet_names = excel_file.sheet_names[:11]

# Combining rates_prices into single df
for i, country_sheet in enumerate(sheet_names):
    # print(i)
    # print(country_sheet)
    new_df = pd.read_excel('G10_RV.xlsx', sheet_name=country_sheet)[['Date', 'Last Price']]
    new_df.columns = ['Date', country_sheet]
    if i == 0:  # Start of the df
        df = new_df
    else:  # Merge to existing df
        df = df.merge(new_df, on='Date', how='outer')

# Filling in missing days with previous observations, defining which columns are rates we want
df = df.set_index('Date')
df = df.resample('D').asfreq()
df = df.ffill()
df = df[::-1].dropna()
rates_tickers = df.columns

def simple(t):
    # print(rates_tickers)
    rates_prices = df[rates_tickers].copy()
    # print(rates_prices)
    
    # Calculating our changes
    for ticker in rates_prices:
        # This line calculates the difference between the current value and the value t periods ahead for each ticker, and stores these differences in a new column named {ticker}_
        rates_prices[f'{ticker}_change_over_t'] = rates_prices[ticker].diff(-t)

    # This line removes any rows from data that contain missing values
    rates_prices = rates_prices.dropna()

    # This line creates a new DataFrame changes that only contains the columns in data that end with '_change_over_t' (the ones that contain the differences calculated earlier)
    changes = rates_prices[[x for x in rates_prices if x.endswith('_change_over_t')]]
    # print(changes)

    # This line splits the changes DataFrame into a training set, which contains all rows with a date index before '2023-1-1'
    changes_training = changes[changes.index < '2023-1-1']

    # This line creates the testing set, which contains all rows with a date index on or after '2023-1-1'
    changes_testing = changes[changes.index >= '2023-1-1']

    result = pd.DataFrame()

    # This line creates a new column 'Predictor' in the result DataFrame, which contains the tickers and some additional strings
    result['Predictor'] = [ticker_to_country[x] for x in rates_tickers] + ['Training r2', 'Testing r2']

    # Building model and storing results for each rate as the target, making df to see results
    for i, target_country in enumerate(changes.columns):
        
        # Stores all the training data in training X, not the current target country, for the X axis
        training_X = changes_training[[x for x in changes_training if x != target_country]]
        
        # Now store the training data of the target country for the Y axis
        training_y = changes_training[target_country]

        # Stores all the testing data in testing X, not the current target country, for the X axis
        testing_X = changes_testing[[x for x in changes_testing if x != target_country]]
        
        # Now store the training data of the target country for the Y axis
        testing_y = changes_testing[target_country]

        # Fit to model
        model = LinearRegression()
        model.fit(training_X, training_y)

        # Prediction over the training dataset
        training_prediction = model.predict(training_X)
        training_r2 = r2_score(training_y, training_prediction)

        # Prediction over the testing dataset
        testing_prediction = model.predict(testing_X)
        testing_r2 = r2_score(testing_y, testing_prediction)

        # Rounding our coefficients 
        coefficients = [round(x, 3) for x in model.coef_]
        # NO Value for prediction against self
        coefficients.insert(i, None)

        # Insert r2 scores
        coefficients.insert(len(coefficients), round(training_r2, 2))
        coefficients.insert(len(coefficients), round(testing_r2, 2))
        
        result[f'y: {ticker_to_country[target_country[:-14]]}'] = coefficients
    result['ABS Mean'] = [result.iloc[i, 1:].dropna().abs().mean() for i in range(11)] + ['', '']
    
    return result

t_values = [1,5,10,25,50,100]

def multi(target, t):
    target_t = f'{target}_change_over_{t}'
    rates_prices = df[rates_tickers].copy()
    # Adding changes for each t value we specified for every ticker
    for ticker in rates_tickers:
        for x in t_values:
            rates_prices[f'{ticker}_change_over_{x}'] = rates_prices[ticker].diff(-x)

    # Training model and storing predictions and performance. Too many predictors to make clean df
    rates_prices = rates_prices.dropna()
    data_training = rates_prices[rates_prices.index < '2023-1-1'].copy()
    data_testing = rates_prices[rates_prices.index >= '2023-1-1'].copy()
    training_X = data_training[[x for x in data_training if '_' in x and x != target_t]]
    training_y = data_training[target_t]
    testing_X = data_testing[[x for x in data_testing if '_' in x and x != target_t]]
    testing_y = data_testing[target_t]
    model = LinearRegression()
    model.fit(training_X, training_y)
    testing_prediction = model.predict(testing_X)
    r2 = round(r2_score(testing_y, testing_prediction),2)
    data_testing['c_prediction'] = testing_prediction
    prediction = data_testing[[target, target_t, 'c_prediction']].copy()
    prediction['prediction'] = prediction[target].shift(-t) + prediction['c_prediction']
    prediction = prediction[[target, 'prediction']].dropna()
    return prediction, r2

def t_eval(t):
    r2s = pd.DataFrame()
    r2s['Target'] = list(country_to_ticker.keys())
    r2s['r2: simple'] = r2s['Target'].apply(lambda x: simple(t)[f'y: {x}'][12])
    r2s['r2: multi'] = r2s['Target'].apply(lambda x: multi(country_to_ticker[x], t)[1])
    r2s = r2s.set_index('Target')
    return r2s

def trader(ticker, t, threshold):
    data = multi(ticker, t)[0]
    data['signal'] = 0
    data.loc[data[ticker] > data['prediction'] + threshold, 'signal'] = -1
    data.loc[data[ticker] < data['prediction'] - threshold, 'signal'] = 1
    data['return'] = data[ticker].diff(-1) * data['signal']
    data = data[::-1]
    data['PnL'] = data['return'].cumsum()
    return data['PnL']

best_values = {}
best_values['US'] = (100, .01)
best_values['Germany'] = (50, .01)
best_values['UK'] = (50, .01)
best_values['France'] = (10, .01)
best_values['Australia'] = (10, .05)
best_values['Canada'] = (100, .01)
best_values['New Zealand'] = (100, .05)
best_values['Japan'] = (10, .01)
best_values['Switzerland'] = (50, .01)
best_values['Norway'] = (50, .01)
best_values['Italy'] = (100, .05)

countries = list(country_to_ticker.keys())
fig, axs = plt.subplots(3, 4, figsize=(12, 9))
axs = axs.flatten()

for i, country in enumerate(countries):
    t, threshold = best_values[country]
    ax = axs[i]
    trade_data = trader(country_to_ticker[country], t, threshold)
    trade_data.plot(ax=ax)
    ax.set_title(country)
    tick_positions = trade_data.index[::100]
    tick_labels = [d.strftime('%b %y') for d in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Performance by Country')
plt.tight_layout()
plt.show()