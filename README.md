# Diamond Price Prediction: Project Overview 
- Created a tool that estimates the price of a  given diamond 
- Engineered two new features, volume and ratio, from the given features
- Optimized a Random Forest regressor using GridSearchCV to reach the model

# Data Cleaning and Feature Engineering
- Replaced any dimensional data with a value of zero with a null value
- Dropped all rows with null values
- Added a new column, Volume, which multiplied the X, Y, and Z dimensional values together
- Made new column, Ratio, which divided the X dimension by the Y dimension
- Added dummy variables for the categorical columns
- Normalized the numerical columns

# Model Building
#### I tried 7 different models and compared them based on Mean Absolute Error ####
- Linear Regression: MAE = 721.9
- Lasso Regression: MAE = 824.8
- Ridge Regression: MAE = 815.9
- Decision Tree: MAE = 356.6
- Random Forest: MAE = 269.2
- Support Vector Machine: MAE = 1059.4
- K Nearest Neighbors: MAE = 445.2
