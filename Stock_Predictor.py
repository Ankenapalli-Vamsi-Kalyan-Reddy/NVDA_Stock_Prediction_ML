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


# Calculate the average of the High and Low prices for each day
df['Avg_High_Low'] = (df['High'] + df['Low']) / 2

# Calculate the daily return percentage change in Adjusted Close price
df['Daily_Return'] = df['Adj Close'].pct_change().fillna(0)

# Display the first few rows of the DataFrame
df.head()

# Uncomment the following lines to calculate the rolling mean and standard deviation
# These can be used to create Bollinger Bands

#rolling_mean = df['Adj Close'].rolling(window=20).mean().fillna(method='bfill')
#rolling_std = df['Adj Close'].rolling(window=20).std().fillna(method='bfill')
#df['Bollinger_High'] = rolling_mean + (rolling_std * 2)
#df['Bollinger_Low'] = rolling_mean - (rolling_std * 2)

# Display the summary information about the DataFrame
df.info()

# Create a column for the next day's Adjusted Close price (for prediction)
df['Next_Close'] = df['Adj Close'].shift(-1)

# Remove any rows with missing values (usually the last row due to shifting)
df.dropna(inplace=True)

# Uncomment the following line to calculate the correlation matrix
# for a specific set of columns, including Open, High, Low, Volume, etc.

#df[['Open', 'High', 'Low','Volume','Range','Hour', 'Minute', 'Next_Close', 'Avg_High_Low']].corr()

# Calculate and display the correlation matrix for the selected columns
df[['Open', 'High', 'Low','Adj Close' ,'Next_Close', 'Avg_High_Low', 'Daily_Return']].corr()

# Set the figure size for the heatmap plot
plt.figure(figsize=(10, 6))

# Uncomment the following lines to plot the heatmap of correlations
# including the Open, High, Low, Volume, Range, etc.

#sns.heatmap(df[['Open', 'High', 'Low', 'Volume', 'Range','Hour', 'Minute', 'Next_Close', 'Avg_High_Low']].corr(), annot=True, cmap='coolwarm')
#sns.heatmap(df[['Open', 'High', 'Low', 'Volume', 'Range','Hour', 'Minute', 'Next_Close', 'Avg_High_Low']].corr(), annot=True, cmap='coolwarm')

# Plot the heatmap of the correlation matrix for the selected columns
sns.heatmap(df[['Open', 'High', 'Low','Adj Close' ,'Next_Close', 'Avg_High_Low', 'Daily_Return']].corr(), annot=True, cmap='coolwarm')

# Display the correlation matrix again (for confirmation)
df[['Open', 'High', 'Low','Adj Close' ,'Next_Close', 'Avg_High_Low', 'Daily_Return']].corr()

# Uncomment the following block to scale the selected features using MinMaxScaler
# This is useful for preparing the data for machine learning models

'''
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the selected features and transform them
scaler.fit(df[['Open', 'High', 'Low','Adj Close', 'Avg_High_Low', 'Daily_Return']])
scaled_features = scaler.transform(df[['Open', 'High', 'Low','Adj Close', 'Avg_High_Low', 'Daily_Return']])

# Create a new DataFrame with the scaled features
scaled_df = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low','Adj Close', 'Avg_High_Low', 'Daily_Return'])

# Display the first few rows of the scaled DataFrame
print(scaled_df.head())
'''

# Uncomment the following block to split the data into training and testing sets

'''
from sklearn.model_selection import train_test_split

# Define the features (X) and the target variable (y)
X = scaled_df
y = df['Next_Close'].reset_index(drop=True)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
'''


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

# Assuming df contains your entire dataset including 'Next_Close'
X = df[['Open', 'High', 'Low', 'Adj Close', 'Avg_High_Low', 'Daily_Return']]
y = df['Next_Close']

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Adj Close', 'Avg_High_Low', 'Daily_Return'])

# Train on the entire dataset
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(scaled_df, y)

# Predict on the entire dataset
df['Predicted_Close'] = knn.predict(scaled_df)

# Save to CSV
df.to_csv('data_with_predictions.csv', index=False)


from sklearn.metrics import mean_absolute_error, mean_squared_error

# For the entire dataset (if you want to evaluate)
predict = df['Predicted_Close']
actual = df['Next_Close']

# Print the Mean Absolute Error (MAE) between the actual and predicted values
print("Mean Absolute Error:", mean_absolute_error(actual, predict))

# Print the Mean Squared Error (MSE) between the actual and predicted values
print("Mean Squared Error:", mean_squared_error(actual, predict))

# Print the Root Mean Squared Error (RMSE) by taking the square root of the MSE
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(actual, predict)))

# Scatter plot of actual vs. predicted
plt.figure(figsize=(10, 6))
plt.scatter(actual, predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

X.info()

# Initialize an empty list to store error rates for different values of K
error_rate = []

# Loop over a range of K values from 1 to 39
for i in range(1, 40):
    # Create a KNeighborsRegressor model with the current K value (number of neighbors)
    model = KNeighborsRegressor(n_neighbors=i)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Predict the target values for the test data
    pred_i = model.predict(X_test)
    
    # Calculate the mean error rate by checking the difference between predictions and actual values
    # Append the calculated error rate to the error_rate list
    error_rate.append(np.mean(pred_i != y_test))

# Create a plot to visualize the error rate against the K values
plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate)
plt.title('Error Rate vs. K Value')  # Set the title of the plot
plt.xlabel('K')  # Label the x-axis as 'K'
plt.ylabel('Error Rate')  # Label the y-axis as 'Error Rate'


# Create a new DataFrame with a single row of stock data for prediction
new_data = pd.DataFrame({
    'Open': [126.4],
    'High': [126.69],
    'Low': [126.38],
    'Adj Close': [126.47],
})

# Calculate the average of the high and low prices and add it as a new column
new_data['Avg_High_Low'] = (new_data['High'] + new_data['Low']) / 2

# (Optional) Calculate the range between high and low prices and add it as a new column
# This line is commented out, but you can uncomment it to include the 'Range' column
# new_data['Range'] = new_data['High'] - new_data['Low']

# Calculate the daily return as the percentage change of the adjusted close price
# Since there is only one row, use fillna(0) to handle the NaN result from pct_change
new_data['Daily_Return'] = new_data['Adj Close'].pct_change().fillna(0)

# Scale the new data using the previously fitted scaler
new_data_scaled = scaler.transform(new_data)

# Use the trained KNN model to predict the target variable (e.g., future stock price) for the new data
prediction = knn.predict(new_data_scaled)


# Create a DataFrame to compare the actual values with the predicted values
comparison_df = pd.DataFrame({
    'Actual': y_test.values,  # Store the actual values from the test set
    'Predicted': predict      # Store the predicted values from the model
})

# Display the first 50 rows of the comparison DataFrame to review the comparison between actual and predicted values
comparison_df.head(50)


comparison_df.corr()




# Create a scatter plot to visualize the relationship between actual and predicted values
plt.scatter(y_test, predict)

# Label the x-axis as 'Actual' to represent the actual values from the test set
plt.xlabel('Actual')

# Label the y-axis as 'Predicted' to represent the predicted values from the model
plt.ylabel('Predicted')

# Set the title of the plot to 'Actual vs Predicted'
plt.title('Actual vs Predicted')

# Display the plot
plt.show()
