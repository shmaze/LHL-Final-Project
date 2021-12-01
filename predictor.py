# import all of our required libraries for necessary data processing and data requests

import numpy as np
import pandas as pd
from binance.client import Client
import joblib



# define our function to retrieve klines data from binance API

def get_data():
    
    '''
    This function will execute API call to Binance to retrieve data.
    We will export the results of this data into the appropriately named dataframe for further feature engineering.
    '''
    
    client = Client()
    # establishing our blank client
    
    candles = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1DAY, limit=90)
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




'''
This now gives us all functions and libraries needed to create our input for the model to predict.
'''

X = to_predict()

# now we must load our saved model using pickle

with open("final_model.pkl", "rb") as file:
    model = joblib.load(file)
    
    
predicted = model.predict_proba(X)




if (predicted[0][0] < predicted[0][1]) & (predicted[0][1] > 0.6):
    print(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go UP tomorrow!\nI am quite confident about this!\nAt this confidence I am correct {53/(53+36)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
    
elif (predicted[0][0] < predicted[0][1]) & (predicted[0][1] > 0.55):
    print(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go UP tomorrow!\nI am sort of confident about this!\nAt this confidence I am correct {36/(36+27)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
              
elif (predicted[0][1] < predicted[0][0]) & (predicted[0][0] > 0.6):
    print(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go DOWN tomorrow!\nI am quite confident about this!\nAt this confidence I am correct {61/(61+44)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
          
elif (predicted[0][1] < predicted[0][0]) & (predicted[0][0] > 0.55):
    print(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go DOWN tomorrow!\nI am sort of confident about this!\nAt this confidence I am correct {38/(38+31)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
          
elif predicted[0][0] < predicted[0][1]:
    print(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go UP tomorrow!\nI am not very confident about this!\nAt this confidence I am correct {67/(67+53)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')

elif predicted[0][1] < predicted[0][0]:
    print(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go DOWN tomorrow!\nI am not very confident about this!\nAt this confidence I am correct {44/(44+36)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
    
else:
    pass
