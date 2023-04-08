# README for BTC prediction model

This code file contains the implementation of a predictive model for Bitcoin (BTC) using historical trading data. The models includes various indicators like:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Rate of Change (ROC)
- Relative Strength Index (RSI)
- Stochastic Oscillator
- Momentum Indicator (MOM)
- Moving Average (MA)

The following machine learning models are used in this implementation.
- Logistic Regression
- Linear Discriminant Analysis
- Decision Tree
- Random Forest

The `BTC_1min_11_22_4.csv` file is the input data source file for this implementation.

The model's purpose is to predict the price movement direction (up or down) by analysing the input data. 

This implementation requires the following python libraries: numpy, pandas, matplotlib, plotly.express, seaborn and sklearn.

The implementation saves the trained model to the file "model_BTC_RF" through pickle.

To perform analysis on the data:
1. Download or clone the project repository.
2. Run the code by importing the required libraries "numpy, pandas, matplotlib, plotly.express, seaborn and sklearn" and run the code block in the code file. 
3. The machine learning model will be trained with the data source file, and a predictive model will be generated and saved to the file "model_BTC_RF".