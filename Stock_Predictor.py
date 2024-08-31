import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the historical NVDA data from multiple CSV files
df1 = pd.read_csv('NVDA_historical_data1_jun10-14.csv')
df2 = pd.read_csv('NVDA_historical_data2_jun17-21.csv')
df3 = pd.read_csv('NVDA_historical_data3_jun24-28.csv')
df4 = pd.read_csv('NVDA_historical_data4_jul1-5.csv')
df5 = pd.read_csv('NVDA_historical_data5_jul8-12.csv')
df6 = pd.read_csv('NVDA_historical_data6_jul15-19.csv')
df7 = pd.read_csv('NVDA_historical_data7_jul22-26.csv')
df8 = pd.read_csv('NVDA_historical_data8_jul29-31.csv')
df9 = pd.read_csv('NVDA_historical_data9_aug1-2.csv')
df10 = pd.read_csv('NVDA_historical_data10_aug5-9.csv')
df11 = pd.read_csv('NVDA_historical_data11_aug12-16.csv')

# Combine all dataframes into a single dataframe
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11])

# Extract the time from the 'Datetime' column as a string (HH:MM)
df['Time'] = df['Datetime'].apply(lambda x: str(x[11:16]))

# Filter rows where 'High' is greater than 150 and less than 1330, then count occurrences
df[(df['High'] > 150) & (df['High'] < 1330)].value_counts()

# Create a new column 'Range' to calculate the difference between 'High' and 'Low'
df['Range'] = df['High'] - df['Low']

# Round the values in the 'Open', 'High', 'Low', and 'Adj Close' columns to 2 decimal places
df['Open'] = df['Open'].round(2)
df['High'] = df['High'].round(2)
df['Low'] = df['Low'].round(2)
df['Adj Close'] = df['Adj Close'].round(2)

# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

# Extract the hour and minute from the 'Time' column and create separate columns
df['Hour'] = df['Time'].apply(lambda x: x.hour)
df['Minute'] = df['Time'].apply(lambda x: x.minute)

# Drop the 'Time' and 'Datetime' columns as they are no longer needed
df.drop(['Time', 'Datetime'], axis=1, inplace=True)

# Display information about the dataframe
df.info()

# Calculate the first quartile (Q1) and third quartile (Q3) for each column
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Calculate lower and upper bounds for detecting outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows that have outliers in any column based on the calculated bounds
df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Display information about the dataframe after removing outliers
df.info()

# Drop the 'Close', 'Volume', 'Range', 'Hour', and 'Minute' columns as they are no longer needed
df.drop(['Close', 'Volume', 'Range', 'Hour', 'Minute'], axis=1, inplace=True)

# Display final information about the dataframe
df.info()
