# LHL-Final-Project
Final Data Science Project for Lighthouse Labs

#### Goals and Aims of this Project
For my final Data Science project at Lighthouse Labs I wanted to create a model that predicts the direction of price movement for Bitcoin. 
I will be using the trading pair BTC/USDT and attempting to use different machine learning classification techniques to determine which type of trading position to open. I will be training this model to attempt to predict whether to open a long trading position (when the price is predicted to rise) or a short trading postiion (when the price is predicted to fall). If this model can predict with an accuracy above 55% it should be able to consistently generate profitable trading positions.

#### Results
Bitcoin is a highly volatile digital asset. It is no stranger to drastic daily and even hourly swings in price, both upward and downward. Throughout the many, many iterations of this project I have gone through so far, the best classification I could achieve was merely 51.9%. 
After this I decided to look a little deeper, and it appeared that a large number of these predictions were being made at a marignal level of confidence (under 55%). Noticing this I decided to look further into the data and was able to find that when measured against a confidence level of 60%, accuracy could be increased to just over 55.39%. 

#### Challenges
With my bare minimum standard hit, I wanted to see if i could improve upon those results. After attmepting to use an LSTM model, and not seeing much improvement due to data I was inputting into the model (garbage in, garbage out!). I realized that I had missed the point of this project, and that was an end to end machine learning process, not what essentially amounted to data analysis of a dataset. 
In realizing I had a lot of ground to make up after wasting much time on the above, I decided to dive in and find a much better data stream using the python-binance library. This made integrating new data easy and seamless, from here I was able to complete much of the same feature engineering from the previous successful sections. From here I was able to train and save a model, which I could then export into a python file which implements functions created to streamline the data intake, feature engineering, and input into the uploaded predictor model.
The script will now output which direction it think the price of bitcoin will move by the end of the next closing day, as well as how confident it is in this prediction. Predictions with over 60% confidence could be used to make trades resulting in profits approximately 50% of the time.