# import all of our required libraries for necessary data processing and data requests

import numpy as np
import pandas as pd
from binance.client import Client
import joblib
import sqlite3
from sqlite3 import Error


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
    
    return X_df





# lets define a function to create our moving averages and incoroprate them into our dataframe

def get_sma(X_df):
    '''
    This function intakes the "X" portion of the data and returns the data with moving average columns applied
    '''
    
    SMAs = [7,30,90]                                                     # 7, 30, and 90 day simple moving averages
    for val in SMAs:
        X_df[str(val)+'sma'] = X_df['Close'].rolling(f'{val}D').mean()   # using the pandas rolling function to calculate mean values over each desired SMA value
        
    return X_df




# Now we want to take the most recent data point possible to make our prediction from

def X_input(X_df):
    x_input = X_df[-1:]        # take the most recent value after calculations for passing into model
    
    return x_input


# now to create a function that ties all of these together and gives us our desired input for the model

def to_predict():
    
    data = get_data()
    data_features = feat_eng(data)
    data_all = get_sma(data_features)
    x_input = X_input(data_all)
    
    return x_input



def add_prediction(X_df):
    pred = model.predict_proba(X_df)[0]                              # this gives us our predictor array of confidence 
    # create our new columns based on prediction output
    X_df['Prediction'] = 1 if pred[1] > pred[0] else 0               # predicted class based on higher confidence
    X_df['Confidence'] = pred[1] if pred[1] > pred[0] else pred[0]   # confidence score (probability) of larger class
    
    return X_df   


def eval_prediction(X_yesterday, X_df):
    '''
    This function will intake our modified X dataframe from the previous day as well as our current prediction
    and output a new column which gives the correct label, as well as if the model predicted correctly or not.
    
    '''
    X_yesterday['True_Label'] = 1 if X_df['Close'].values > X_yesterday['Close'].values else 0                         # this gives us the correct label
    X_yesterday['Correct_Pred'] = 1 if X_yesterday['Prediction'].values == X_yesterday['True_Label'].values else 0      # this gives a 1 for a correct prediction and a 0 for incorrect
    
    return X_yesterday


# creating a function which will pull the previous days data instead of today 

def X_input_yesterday(X_df):
    x_input = X_df[-2:-1]        # take the most recent value after calculations for passing into model
    
    return x_input


# now to create a function that ties all of these together and gives us our desired input for the model

def to_predict_yesterday():
    
    data = get_data()
    data_features = feat_eng(data)
    data_all = get_sma(data_features)
    x_input_yesterday = X_input_yesterday(data_all)
    
    return x_input_yesterday


# define a new function to intake our X_evaluated dataframe and write it to our new 
# SQL database for future use

def to_database(X_evaluated):
    '''
    This function takes in our fully evaluated predictions and writes them 
    to an SQL database for further reference.
    '''
    X_evaluated = X_evaluated.sort_index()
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
        
    X_evaluated.to_sql('predictions', con=conn, if_exists='append')
    
        
    conn.close()
    
    
# Now that we have imported all of the necesary functions, we can incorporate our process of evaluation.


# now we must load our saved model using pickle

with open("final_model.pkl", "rb") as file:
    model = joblib.load(file)
    
    
    



# we also want to update our model metrics database at the same time, we will import the necessary functions

def from_database():
    '''
    This function retrieves our fully evaluated predictions from
    our SQL database for further reference.
    '''
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
        
    query = """
            SELECT Date, 
            Close, 
            Prediction, 
            Confidence, 
            True_Label, 
            Correct_Pred 
            FROM predictions
            """
    
    data = pd.read_sql(query, conn, parse_dates=['Date'])        # adding in the query, connection, and parsing date column as its correct format
    data.set_index('Date', inplace=True)                         # setting the index of the resulting dataframe to the date column
    conn.close()                                                 # closiung connection to database
    
    return data.sort_index(inplace=True)




# now we need to establish model accuracy measure
# this will be the sum of the correct_pred column divided by its length
# to do this we will calculate the metric each time the new data is imported and append it to a new column in the dataframe
# after this new column is created we will establish a new database table which only stores our model metrics


def model_metrics(X_from_db):
    '''
    This function will take in our queried dataframe and add a new column called model accuracy.
    This will be updated daily as each new prediction is implemented.
    '''
    X_from_db['Model_Accuracy'] = 0
    
    model_acc = sum(X_from_db['Correct_Pred']) / len(X_from_db['Correct_Pred'])
    X_from_db['Model_Accuracy'][-1:] = model_acc * 100
    
    return X_from_db.round(2)





# define a new function to intake our X_evaluated dataframe and write it to our new 
# SQL database for future use

def to_acc_db(X_from_db):
    '''
    This function takes in our model performance metrics dataframe and writes it to a new table in our database.
    '''
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
        
    X_from_db[-1:].to_sql('performance', con=conn, if_exists='append')
    
        
    conn.close()


    
    
# we will slightly modify a function used in previous sections to access our desired columns only


def get_trade_info(limit=1):
    '''
    This will retrieve our most recent prediction
    '''
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
        
    if limit==None:
        query = """
            SELECT Date, Close as Entry, Correct_Pred as Win
            FROM performance ORDER BY Date DESC
            """
    else:
        query = f"""
            SELECT Date, Close as Entry, Correct_Pred as Win
            FROM performance ORDER BY Date DESC LIMIT {limit}
            """
    
    data = pd.read_sql(query, conn, parse_dates=['Date'])        # adding in the query, connection, and parsing date column as its correct format
    data.set_index('Date', inplace=True)                         # setting the index of the resulting dataframe to the date column
    conn.close()                                                 # closiung connection to database
    
    return data



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



# now to create a function that will calcuate our net percentage difference in price change and assign a value against 
# todays price which will be fetched from binance


# now we need a function to recall the most recent stake out value to be input into our other function to calculate
# our cumulative gains

def get_stake():
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
    
    query = """
            SELECT Stake_Out
            FROM quantitative_stats
            ORDER BY Date DESC LIMIT 1
            """
    
    data = pd.read_sql(query, conn, parse_dates=['Date'])
    
        
    conn.close()
    
    return data.Stake_Out.values[0]


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



# now we need one last function which will write our new data into our last database

def gains_db(gains_df):
    '''
    This function will take in our created gains df and append it to our existing database.
    '''
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Saved to Database!')
    
    except Error as e:
        print(e)
        
    gains_df.to_sql('quantitative_stats', con=conn, if_exists='append')
    
        
    conn.close()
    
# this function will create csv files from all of our databases as well
    
def make_csvs(X_evaluated, X_from_db, gains_df):
    '''
    This function intakes our fully completed dataframes from each section of our database populator and writes them to csv files.
    '''
    X_evaluated.to_csv('./CSVs/predictions.csv', index_label='Date')
    X_from_db.to_csv('./CSVs/performance.csv', index_label='Date')
    gains_df.to_csv('./CSVs/quantitative_stats.csv', index_label='Date')
    
    print("CSV's created!")  

    
    
    
today = to_predict()
yesterday = to_predict_yesterday()
yesterday_predicted = add_prediction(yesterday)
X_yesterday = eval_prediction(yesterday_predicted, today)
to_database(X_yesterday)

data = from_database()
X_from_db = model_metrics(data)
to_acc_db(X_from_db)

gains_df = get_gains()
gains_db(gains_df)

print('Databases updated!') 
make_csvs(X_yesterday, X_from_db, gains_df)

# this will retrieve and calculate all necessary metrics and write them to our database for tracking    



  







