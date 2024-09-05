# KNN-Based Stock Prediction with Power BI Integration

This project demonstrates the integration of a K-Nearest Neighbours (KNN) regression model for stock price prediction into Power BI. The model is applied to historical NVDA stock data, with a focus on enhancing data insights and visualizing predictions.

## Overview
- **Model**: K-Nearest Neighbours (KNN) Regression
- **Data**: Historical NVDA stock data
- **Metrics**:
  - Mean Absolute Error (MAE): 0.285
  - Mean Squared Error (MSE): 0.221
  - Root Mean Squared Error (RMSE): 0.470

## Features
- **Dynamic Filters**: 
  - Implemented 3 dynamic filters using slicers and donut charts to analyze prediction accuracy and threshold variances.
  
- **Visualizations**:
  - Used over 4 charts and KPIs, including scatterplots, line charts, and cards, to compare actual vs. predicted stock prices.

## Workflow
1. **Data Preprocessing**:
   - Combined multiple CSV files of NVDA historical data.
   - Extracted and transformed relevant features.
   - Removed outliers using Interquartile Range (IQR) method.
   
2. **Feature Engineering**:
   - Calculated the average of high and low prices.
   - Computed the daily return percentage change in adjusted close price.
   - Created a target variable (`Next_Close`) for prediction.

3. **Model Training and Evaluation**:
   - Trained a KNN model on scaled features.
   - Evaluated model performance using MAE, MSE, and RMSE.

4. **Power BI Integration**:
   - Integrated the model predictions into Power BI.
   - Visualized predictions and analysis using Power BI charts and KPIs.

## Results
- Achieved a high level of accuracy with the KNN model, with low error metrics.
- Effectively visualized prediction results and insights using Power BI.

## Files Included
- `Stock_NVDA.pbix`: Power BI file with integrated KNN model and visualizations.
- `data_with_predictions.csv`: CSV file containing the dataset with predicted values.
- `KNN_Stock_Prediction.py`: Python script used for data processing and model training.

## Usage
- Clone the repository and open the `.pbix` file in Power BI to explore the visualizations.
- Review the Python script to understand the data processing and model training steps.

## Future Work

- **Model Improvement:** Explore advanced machine learning models like LSTM or XGBoost for better accuracy.
- **Data Sources:** Incorporate additional data sources such as financial news or social media sentiment to enhance predictions.
- **Real-time Integration:** Implement real-time data fetching and prediction in Power BI for live stock analysis.


