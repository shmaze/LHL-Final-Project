# import all of our required libraries for necessary data processing and data requests

import numpy as np
import pandas as pd
from binance.client import Client
import joblib
import os




# define our function to retrieve klines data from binance API

def get_data():
    
    '''
    This function will execute API call to Binance to retrieve data.
    We will export the results of this data into the appropriately named dataframe for further feature engineering.
    '''
    
    client = Client()
    # establishing our blank client
    
    candles = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY, limit=91)
    # we only need to request the most recent 90 days to calculate our prediction data
    
    data = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore'])
    # these column labels are as labelled on the Binance API documentation
    
    data.drop(['Close time', 'Ignore'], axis=1, inplace=True)
    # dropping unneeded columns
    
    data['Date'] = data['Date'].apply(lambda x: pd.to_datetime(x, unit='ms'))
    # converting to proper date format for better visual reference
    
    data.set_index('Date', inplace=True)
    # setting index to date
    
    data = data.astype('float64')
    # converting from object type to float type
    
    return data





# we will define a function to run prior to calcualting our averages

def feat_eng(X_df):
    '''
    Intakes "X" portion of data and outputs selected engineered features
    '''
    
    X_df['High/Low'] = X_df['High'] - X_df['Low']
    X_df['volX'] = X_df['Quote asset volume'] / X_df['Volume']
    X_df['quote-buy'] = X_df['Taker buy quote volume'] / X_df['Taker buy base volume']

    SMAs = [7,30,90]                                                     # 7, 30, and 90 day simple moving averages
    for val in SMAs:
        X_df[str(val)+'sma'] = X_df['Close'].rolling(f'{val}D').mean()   # using the pandas rolling function to calculate mean values over each desired SMA value
    
    return X_df



# Now we want to take the most recent data point possible to make our prediction from

def X_inputs(X_df):
    x_input = X_df[-1:]           # take the most recent value after calculations
    x_yesterday = X_df[-2:-1]     # values from previous day
    
    return x_input, x_yesterday


# now to create a function that ties all of these together and gives us our desired inputs for the model

def to_predict():
    
    data = get_data()
    X_df = feat_eng(data)
    X_input, X_yesterday = X_inputs(X_df)
    
    return X_input, X_yesterday


# now we must load our saved model using pickle

with open("final_model.pkl", "rb") as file:
    model = joblib.load(file)  
    
    

def add_prediction(X_input, X_yesterday):
    pred_X = model.predict_proba(X_input)[0]                              # this gives us our predictor array of confidence 
    # create our new columns based on prediction output
    X_input['Prediction'] = 1 if pred_X[1] > pred_X[0] else 0               # predicted class based on higher confidence
    X_input['Confidence'] = pred_X[1] if pred_X[1] > pred_X[0] else pred_X[0]   # confidence score (probability) of larger class
    
    pred_yesterday = model.predict_proba(X_yesterday)[0]
    X_yesterday['Prediction'] = 1 if pred_yesterday[1] > pred_yesterday[0] else 0               
    X_yesterday['Confidence'] = pred_yesterday[1] if pred_yesterday[1] > pred_yesterday[0] else pred_yesterday[0]
    
    return X_input, X_yesterday   


def eval_prediction(X_input, X_yesterday):
    '''
    This function will intake our modified X dataframe from the previous day as well as our current prediction
    and output a new column which gives the correct label, as well as if the model predicted correctly or not.
    
    '''
    X_yesterday['True_Label'] = 1 if X_input['Close'].values > X_yesterday['Close'].values else 0                         # this gives us the correct label
    X_yesterday['Correct_Pred'] = 1 if X_yesterday['Prediction'].values == X_yesterday['True_Label'].values else 0      # this gives a 1 for a correct prediction and a 0 for incorrect
    
    return X_yesterday



# in this version we will be moving away from using sql to store our data and instead using csv files which are convenient and easy to export to google drive or other services. 

def to_predictions_csv(X_yesterday):
    '''
    This function takes in our fully evaluated predictions and writes them 
    to a CSV file.
    '''
    if os.path.isfile('./CSVs/model_predictions.csv'):
        X_yesterday.to_csv('./CSVs/model_predictions.csv', mode='a', header=0)
    else:
        X_yesterday.to_csv('./CSVs/model_predictions.csv')
    print('Data written to model_predictions.csv!')
    
    
# Now that we have imported all of the necesary functions, we can incorporate our process of evaluation
    
    
# we will now also want to draw on the full CSV file to calculate our ongoing model metrics.
# now we need to establish model accuracy measure
# this will be the sum of the correct_pred column divided by its length
# to do this we will calculate the metric each time the new data is imported and append it to a new column in the dataframe

def get_performance():
    
    '''
    This function will take in our model performance CSV and add a new column called model accuracy.
    This will be updated daily as each new prediction is implemented. We will take only the most recent value as an output so that we only add the most recent row to our next CSV.
    '''
    model_data = pd.read_csv('./CSVs/model_predictions.csv', parse_dates=['Date'])
    model_data.set_index('Date', inplace=True)
    model_data['Model_Accuracy'] = 0
    
    model_acc = sum(model_data['Correct_Pred']) / len(model_data['Correct_Pred'])
    model_data['Model_Accuracy'][-1:] = model_acc * 100
    model_data = model_data.filter(['Date', 'Close', 'Prediction', 'Confidence', 'True_Label', 'Correct_Pred', 'Model_Accuracy'], axis=1)
    model_data = model_data.round(2)
    for_CSV_data = model_data[-1:]
    
    return for_CSV_data


# now we will write our model performance measures to another CSV

def to_performance_csv(for_CSV_data):
    '''
    This function takes in our fully evaluated predictions and writes them 
    to a CSV file.
    '''
    if os.path.isfile('./CSVs/model_performance.csv'):
        for_CSV_data.to_csv('./CSVs/model_performance.csv', mode='a', header=0)
    else:
        for_CSV_data.to_csv('./CSVs/model_performance.csv')
    
    print('Data written to model_performance.csv!')



# we will access our most recent row in our performance CSV column to get our desired trade info and calculate our current quantitative stats

def get_trade_info():
    
    trade_info = pd.read_csv('./CSVs/model_performance.csv', parse_dates=['Date'])
    trade_info.set_index('Date', inplace=True)
    trade_info = trade_info[-1:]
    trade_info = trade_info.rename({'Close':'Entry', 'Correct_Pred':'Win'}, axis=1)
    trade_info = trade_info.filter(['Date', 'Entry', 'Win'], axis=1)
    return trade_info




# we will slightly modify our data gathering function from our predictor script

def get_price():
    
    '''
    This function will execute API call to Binance to retrieve data.
    We will export the results of this data into the appropriately named dataframe for further feature engineering.
    '''
    
    client = Client()
    # establishing our blank client
    
    candles = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY, limit=1)
    # we only need to request the most recent entry
    
    data = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base volume', 'Taker buy quote volume', 'Ignore'])
    # these column labels are as labelled on the Binance API documentation
    
    price = data[['Date', 'Close']]
    # taking only the desired columns
    
    price['Date'] = price['Date'].apply(lambda x: pd.to_datetime(x, unit='ms'))
    # setting date to proper format
    
    price.set_index('Date', inplace=True)
    #setting our index
    
    
    return price



# we will need a function to retrieve our current stake price, as we are stsarting out initially, we will not have a stake in any data
# we will creat a function that will check if our desired CSV files exists tracking our stake
# if it does not exist, the function will use our inital stake of 1000

def get_stake():
    
    if os.path.isfile('./CSVs/model_quantitative_stats.csv'):
        data = pd.read_csv('./CSVs/model_quantitative_stats.csv', parse_dates=['Date'])
        data.set_index('Date', inplace=True)
        data_needed = data[-1:]
        stake = data_needed['Stake_Out'].value
        
        
    else:
        stake = 1000
        
    return stake
    
    


# now to create a function that will calcuate our net percentage difference in price change and assign a value against 
# todays price which will be fetched from binance




def get_gains():
    '''
    This function will intake the resulting dataframes from the get_trade_info  and get_price functions and check the result against 
    our entry price. This will then calculate a net percentage change. Using the value in the "win" column,
    we can tell the function to apply a positive or negative change to our stake value. All of this data will then
    be stored to a new database for recall.
    '''
    trades = get_trade_info()
    price = get_price()
    price.Close = price.Close.astype('float')
    stake_in = get_stake()
    
    
    trades['Exit'] = price.Close.values
    net_change = abs(trades.Exit - trades.Entry)
    trades['Pct_Change'] = 0
    trades['Gains(%)'] = 0
    trades['Stake_In'] = stake_in
    trades['Stake_Out'] = 0
    trades['Net_Profits'] = 0
    trades['Profit_YTD'] = 0
    trades['ROI(%)'] = 0
    
    
    pct = net_change / trades.Entry
    trades['Pct_Change'] = pct
    trades['Gains(%)'] = -(trades.Pct_Change) if trades.Win.values == 0 else trades.Pct_Change
    trades['Stake_Out'] = trades.Stake_In + (trades.Stake_In * trades['Gains(%)'])
    trades['Net_Profits'] = trades.Stake_Out - trades.Stake_In
    trades['Profit_YTD'] = trades.Stake_Out - 1000
    trades['ROI(%)'] = trades.Profit_YTD / 1000
    
    return trades


# now we just need to write our final data to a model quantitative stats CSV

def to_quantitative_csv(trades):
    '''
    This function takes in our fully evaluated trades info and writes it to our last CSV file.
    '''
    if os.path.isfile('./CSVs/model_quantititive_stats.csv'):
        trades.to_csv('./CSVs/model_quantititive_stats.csv', mode='a', header=0)
    else:
        trades.to_csv('./CSVs/model_quantititive_stats.csv')
    
    print('Data written to model_quantitative_stats.csv!')




X_input, X_yesterday = to_predict()
X_input, X_yesterday = add_prediction(X_input, X_yesterday)
X_yesterday = eval_prediction(X_input, X_yesterday)

to_predictions_csv(X_yesterday)
for_CSV_data = get_performance()
to_performance_csv(for_CSV_data)

trades = get_gains()
to_quantitative_csv(trades)

print('CSVs updated successfully!')

# this works great, now we just need to schedule this script to run on a daily basis, and then incorporate it exporting our CSVs to google for cloud app deployment.













