import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from binance.client import Client
import joblib
from urllib.request import urlopen


# this function will retrieve our stored csv files

def get_csvs():
    
    predictions = pd.read_csv('https://raw.githubusercontent.com/shmaze/LHL-Final-Project/main/CSVs/model_predictions.csv', index_col='Date', parse_dates=True)
    
    performance = pd.read_csv('https://raw.githubusercontent.com/shmaze/LHL-Final-Project/main/CSVs/model_performance.csv', index_col='Date', parse_dates=True)
    
    quant_stats = pd.read_csv('https://raw.githubusercontent.com/shmaze/LHL-Final-Project/main/CSVs/model_quantitative_stats.csv', index_col='Date', parse_dates=True)
    return predictions, performance, quant_stats


# define our functions to retrieve klines data from binance API for live predictor

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





#This now gives us all functions and libraries needed to create our input for the model to predict.


X = to_predict()

# now we must load our saved model using joblib

model_url = 'https://drive.google.com/file/d/1hQURz8A2l3iqc6y2HJHv6ZBZ0qwm8KVM/view?usp=sharing'
model_path = 'https://drive.google.com/uc?export=download&id=' + model_url.split('/')[-2]
model = joblib.load(urlopen(model_path))
    
predicted = model.predict_proba(X)

preds, perf, quant = get_csvs()


# now that we have set all of our parameters out, we can start to set our streamlit layout

# we will now define a function that takes in our prediction and gives us an output based on its result

def print_pred(predicted):
    if (predicted[0][0] < predicted[0][1]) & (predicted[0][1] > 0.6):
        st.write(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go UP tomorrow!\nI am quite confident about this!\nAt this confidence I am correct {53/(53+36)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
    
    elif (predicted[0][0] < predicted[0][1]) & (predicted[0][1] > 0.55):
        st.write(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go UP tomorrow!\nI am sort of confident about this!\nAt this confidence I am correct {36/(36+27)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
              
    elif (predicted[0][1] < predicted[0][0]) & (predicted[0][0] > 0.6):
        st.write(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go DOWN tomorrow!\nI am quite confident about this!\nAt this confidence I am correct {61/(61+44)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
          
    elif (predicted[0][1] < predicted[0][0]) & (predicted[0][0] > 0.55):
        st.write(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go DOWN tomorrow!\nI am sort of confident about this!\nAt this confidence I am correct {38/(38+31)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')
          
    elif predicted[0][0] < predicted[0][1]:
        st.write(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go UP tomorrow!\nI am not very confident about this!\nAt this confidence I am correct {67/(67+53)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')

    elif predicted[0][1] < predicted[0][0]:
        st.write(f'*********************\n\n\n\nThe price of Bitcoin is predicted to go DOWN tomorrow!\nI am not very confident about this!\nAt this confidence I am correct {44/(44+36)*100:.2f}% of the time!\n\n\n\nThis is not finanical advice. I am not a financial advisor. All information here is for entertainment purposes only.\n\n\n\n*********************')




st.title('Bitcoin Price Direction Movement Predictor and Performance')


def display_model_performance(model_metrics_df, quant_data):
    '''
    This function takes in the pandas dataframe created when retrieving accuracy metrics from associated dataframe.
    It returns total number of predictions made, total number of correct predictions,
    number if correct predictions per class, model accuracy over time.
    
    This function will output dsiplays and graphs to showcase this data.
    '''
    
    
    # first step is to set all variables to be displayed in our metrics
    total_predictions = len(model_metrics_df)
    total_1_pred = sum(model_metrics_df.Prediction)
    total_0_pred = total_predictions - total_1_pred
    total_correct = sum(model_metrics_df.Correct_Pred)
    current_acc = model_metrics_df.Model_Accuracy[-1:]
    
    correct_only = model_metrics_df[model_metrics_df['Prediction'] == model_metrics_df['True_Label']]
    correct_1 = sum(correct_only.True_Label)
    correct_0 = len(correct_only) - correct_1
    acc_1 = correct_1 / total_1_pred * 100
    if total_0_pred == 0:
        acc_0 = 100.0
    else:
        acc_0 = correct_0 / total_0_pred * 100
    
   
    
    fig, (ax1, ax2, ax3, ax4, ax5)  = plt.subplots(5,1, figsize=(15,20))
    
    ax1.bar([0,1], [total_0_pred, total_1_pred] , color='royalblue', width=0.4,label='Predicted')
    ax1.bar([0+0.4,1+0.4], [correct_0, correct_1], color='seagreen', width=0.4, label='Correct')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(labels=[0,1], fontsize=20)
    ax1.set_xlabel('Predicted Direction', fontsize=25)
    ax1.set_title('Predicted vs. Correct for Each Class', fontsize=25)
#     ax1.set_yticks(range(0, total_predictions, 1))
#     ax1.set_yticklabels(range(0, total_predictions, 1), fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_ylabel('Count', fontsize=20)
    ax1.legend(fontsize=15)
                   
    
    ax2.plot(model_metrics_df.index, model_metrics_df.Model_Accuracy, label='Accuracy (%)', marker='o')  
    ax2.set_title('Model Accuracy Over Time', fontsize=25)
    ax2.set_yticks(range(0,110,10))
    ax2.set_ylabel('Model Accuracy (%)', fontsize=20)
    ax2.set_xlabel('Date', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    
    
    ax3.plot(quant_data.index, quant_data['Gains(%)'], color='royalblue', label='Daily Gain(%)', marker='o')
    ax3.plot(quant_data.index, quant_data['ROI(%)'], color='seagreen', label='Model ROI(%)', marker='x')
    ax3.set_xlabel('Date', fontsize=25)
    ax3.set_title('Daily Gains and Model Return on Investment', fontsize=25)
    ax3.set_ylabel('Return', fontsize=20)
    ax3.legend(fontsize=15)
    ax3.tick_params(axis='both', which='major', labelsize=15)              
    
    ax4.plot(quant_data.index, quant_data['Net_Profits'], label='Daily', marker='o', color='royalblue')  
    ax4.plot(quant_data.index, quant_data['Profit_YTD'], label='Year-to-Date', marker='x', color='seagreen')
    ax4.set_title('Daily and Year-to-Date Net Profits', fontsize=25)
    ax4.set_ylabel('Profit($)', fontsize=20)
    ax4.set_xlabel('Date', fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=15)
    ax4.legend(fontsize=15)
    
    ax5.plot(quant_data.index, quant_data['Stake_Out'], label='Total Value', color='royalblue', marker='x')
    ax5.set_title('Total Value Over Time', fontsize=25)
    ax5.set_ylabel('Total Value($)', fontsize=20)
    ax5.set_xlabel('Date', fontsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=15)
    
    fig.tight_layout(pad=3)
    
    
    metrics = pd.DataFrame({'Predictions': total_predictions,
                            'Correct': total_correct,
                            'Live Accuracy': current_acc,
                            'Up Accuracy': acc_1,
                            'Down Accuracy': acc_0
                           })
    return metrics.round(2), fig

metrics, fig = display_model_performance(perf, quant)

print_pred(predicted)

st.table(metrics)

st.pyplot(fig)



