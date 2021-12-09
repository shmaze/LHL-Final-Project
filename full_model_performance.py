# import necessary libraries

import sqlite3
import pandas as pd
import numpy as np
from binance.client import Client
import joblib
from sqlite3 import Error
import matplotlib.pyplot as plt

# now to define a new function to retrieve our desired data for model metrics


def from_acc_db():
    '''
    This function recalls our model metrics database and plots various values and metrics for model performance over time.
    '''
    
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_model.db')
        print('Connected Successfully!')
    
    except Error as e:
        print(e)
        
    query = """
            SELECT *
            FROM performance
            """
    
    data = pd.read_sql(query, conn, parse_dates=['Date'])        # adding in the query, connection, and parsing date column as its correct format
    data.set_index('Date', inplace=True)                         # setting the index of the resulting dataframe to the date column
    conn.close()                                                 # closiung connection to database
    
    return data



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






# now we want to establish what we would like to display from our model metrics
# date vs accuracy is obviously important
# we also want to show price vs time and some way to show amount to correct predictions


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
    
    correct_only = model_metrics[model_metrics['Prediction'] == model_metrics['True_Label']]
    correct_1 = sum(correct_only.True_Label)
    correct_0 = len(correct_only) - correct_1
    acc_1 = correct_1 / total_1_pred * 100
    if total_0_pred == 0:
        acc_0 = 100.0
    else:
        acc_0 = correct_0 / total_0_pred
    
   
    
    fig, (ax1, ax2, ax3, ax4, ax5)  = plt.subplots(5,1, figsize=(15,20))
    
    ax1.bar([0,1], [total_0_pred, total_1_pred] , color='royalblue', width=0.4,label='Predicted')
    ax1.bar([0+0.4,1+0.4], [correct_0, correct_1], color='seagreen', width=0.4, label='Correct')
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(labels=[0,1], fontsize=20)
    ax1.set_xlabel('Predicted Direction', fontsize=25)
    ax1.set_title('Predicted vs. Correct for Each Class', fontsize=25)
    ax1.set_yticks(range(0, total_predictions, 1))
    ax1.set_yticklabels(range(0, total_predictions, 1), fontsize=20)
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
    fig.show()
    
    metrics = pd.DataFrame({'Predictions': total_predictions,
                            'Correct': total_correct,
                            'Live Accuracy': current_acc,
                            'Up Accuracy': acc_1,
                            'Down Accuracy': acc_0
                           })
    return metrics.round(2)

# now we run both functions inputting results of first into the second. 
# the output will be a dataframe showing current model metrics and 2 graphs showing model performance.


model_metrics = from_acc_db()
quant_data = get_quant_data()
print(display_model_performance(model_metrics, quant_data))
