# LHL-Final-Project
Final Data Science Project for Lighthouse Labs


##### Final Product is Deployed. Check it out below!

[Check out the model!!!](https://share.streamlit.io/shmaze/lhl-final-project/main/predictor_app.py)





#### Goals and Aims of this Project
For my final Data Science project at Lighthouse Labs I wanted to create a model that predicts the direction of price movement for Bitcoin. 

I will be using the trading pair BTC/USDT and attempting to use different machine learning classification techniques to determine which type of trading position to open. I will be training this model to attempt to predict whether to open a long trading position (when the price is predicted to rise) or a short trading postiion (when the price is predicted to fall). If this model can predict with an accuracy above 55% it should be able to consistently generate profitable trading positions.


#### Contents
[Chapter 1](https://github.com/shmaze/LHL-Final-Project/tree/main/1-Notebooks) - These are a collection of failed attempts. Valuable for learning what not to do.

[Chapter 2](https://github.com/shmaze/LHL-Final-Project/blob/main/2-1_Backtesting%20and%20Re-evaluating.ipynb) - This was a detour and waste of time. Although interesting for future exploration.

[Chapter 3](https://github.com/shmaze/LHL-Final-Project/blob/main/3-1_Recurrent-Neural-Networks-and-LSTM.ipynb) - This was the foray into an LSTM network. Less than fantastic results.

[Chapter 4-1](https://github.com/shmaze/LHL-Final-Project/blob/main/4-1_Starting-Over.ipynb) - This is where all the magic happened. Results included here. Final model exported at the end.

[Chapter 4-2](https://github.com/shmaze/LHL-Final-Project/blob/main/4-2_Starting-the-Framework-for-the-Program.ipynb) - This is the notebook used to create all the functions exported into predictor.py script.

[Chapter 5](https://github.com/shmaze/LHL-Final-Project/blob/main/5_Evaluations-and-Tables.ipynb) - Simple dataframe for presentation of successful model metrics.

Follow-up Chapters

Chapter 6 - These notebooks contain all processes related to model tracking and performance.

  ~[6-1](https://github.com/shmaze/LHL-Final-Project/blob/main/6-1_Model-Tracking-and-Database-Creation.ipynb) - Evaluating model performance metrics and writing to our first database.
  
  ~[6-2](https://github.com/shmaze/LHL-Final-Project/blob/main/6-2_Reading-New-Database-and-Visualizing-Model-Performance.ipynb) - Reading data from our new database and displaying model performance with matplotlib.
  
  ~[6-3](https://github.com/shmaze/LHL-Final-Project/blob/main/6-3_Implementing-Quantitative-Metrics.ipynb) - Implementing quantitative model tracking metrics and improving upon prior model performance display.
  
Chapter 7 - Notebooks for testing and creating new CSV files and Streamlit app deployment 

  ~[7-1](https://github.com/shmaze/LHL-Final-Project/blob/main/7-1_Attempting-Google-cloud-CSV-Storage-and-Streamlit-Deployment.ipynb) - First steps into testing google cloud storage and deployment.
  
  ~[7-2](https://github.com/shmaze/LHL-Final-Project/blob/main/7-2_Re-Working-for-Deployment.ipynb) - Created functions for accessing and stroing csvs.
  
[Deployable App](https://github.com/shmaze/LHL-Final-Project/blob/main/predictor_app.py) - This is the final script which is deployed onto Streamlit and is the finished product as this time.
  
Scripts - Final runnable python scripts for [Predicting](https://github.com/shmaze/LHL-Final-Project/blob/main/predictor_app.py) and [Database_Population](https://github.com/shmaze/LHL-Final-Project/blob/main/final_database_populator.py). As well as [deprecated scripts](https://github.com/shmaze/LHL-Final-Project/tree/main/Deprecated-Scripts) that have been improved upon and are no longer needed.

### Disclaimer
### This is not financial advice. I am not a finanical advisor. Everything contained herein is meant to be for entertainment purposes only. 

#### Results
Bitcoin is a highly volatile digital asset. It is no stranger to drastic daily and even hourly swings in price, both upward and downward. Throughout the many, many iterations of this project I have gone through so far, the best classification I could achieve was merely 51.9%. 

After this I decided to look a little deeper, and it appeared that a large number of these predictions were being made at a marignal level of confidence (under 55%). Noticing this I decided to look further into the data and was able to find that when measured against a confidence level of 60%, accuracy could be increased to just over 55.39%. 

#### Challenges
With my bare minimum standard hit, I wanted to see if i could improve upon those results. After attmepting to use an LSTM model, and not seeing much improvement due to data I was inputting into the model (garbage in, garbage out!). I realized that I had missed the point of this project, and that was an end to end machine learning process, not what essentially amounted to data analysis of a dataset. 

In realizing I had a lot of ground to make up after wasting much time on the above, I decided to dive in and find a much better data stream using the [python-binance](https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data) library. This made integrating new data easy and seamless, from here I was able to complete much of the same feature engineering from the previous successful sections. From here I was able to train and save a model, which I could then export into a python file which implements functions created to streamline the data intake, feature engineering, and input into the uploaded predictor model.

The script will now output which direction it think the price of bitcoin will move by the end of the next closing day, as well as how confident it is in this prediction. Predictions with over 60% confidence could be used to make trades resulting in profits approximately 58.5% of the time.

#### Final Results


![alt text](https://github.com/shmaze/LHL-Final-Project/blob/main/data/accuracy_chart.png)
