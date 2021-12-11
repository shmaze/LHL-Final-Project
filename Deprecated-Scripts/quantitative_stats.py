import numpy as np
import pandas as pd
from binance import Client
import matplotlib.pyplot as plt
import sqlite3
from sqlite3 import Error


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
    
    

    
# now we will set out the steps to be run by the script upon initialization


gains_db(get_gains())

# this will retrieve and calculate all necessary metrics and write them to our database for tracking



# now we also want to be able to display all of these metrics in a nice format

# we will use matplotlib to visualize our modls performance over time, quantifying things like ROI and overall profit
# we will also want to use the most up to date metrics for display, so we will also define a function to recall all data in our desired format for charting


# a function for retrieving all relevant information for displayiong model quantitative performance over time


def get_quant_data():
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
    
    query = r"""
            SELECT *
            FROM quantitative_stats
            ORDER BY Date
            """
    
    data = pd.read_sql(query, conn, parse_dates=['Date'])
    
    data.set_index('Date', inplace=True)
    data.drop(['Entry', 'Exit', 'Win', 'Pct_Change', 'Stake_In'], axis=1, inplace=True)
    conn.close()
    
    return data



# now we need a function that will output all of our desired metrics


def quant_charts(quant_data):
    '''
    This function intakes the resulting dataframe of the "get_quant_data" function and outputs graphs and charts showing 
    model performance over time
    '''
   
    fig, (ax1, ax2, ax3)  = plt.subplots(3,1, figsize=(15,9))
    
    ax1.plot(quant_data.index, quant_data['Gains(%)'], color='royalblue', label='Daily Gain(%)', marker='o')
    ax1.plot(quant_data.index, quant_data['ROI(%)'], color='seagreen', label='Model ROI(%)', marker='x')
    ax1.set_xlabel('Date', fontsize=25)
    ax1.set_title('Daily Gains and Model Return on Investment', fontsize=25)
    ax1.set_ylabel('Return', fontsize=20)
    ax1.legend(fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)              
    
    ax2.plot(quant_data.index, quant_data['Net_Profits'], label='Daily', marker='o', color='royalblue')  
    ax2.plot(quant_data.index, quant_data['Profit_YTD'], label='Year-to-Date', marker='x', color='seagreen')
    ax2.set_title('Daily and Year-to-Date Net Profits', fontsize=25)
    ax2.set_ylabel('Profits', fontsize=20)
    ax2.set_xlabel('Date', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.legend(fontsize=15)
    
    ax3.plot(quant_data.index, quant_data['Stake_Out'], label='Total Value', color='royalblue', marker='x')
    ax3.set_title('Total Value Over Time', fontsize=25)
    ax3.set_ylabel('Total Value', fontsize=20)
    ax3.set_xlabel('Date', fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    
    fig.tight_layout(pad=3)
    fig.show()
    

    
    
# now we will combine these functions to output our desired model metrics

quant_charts(get_quant_data())