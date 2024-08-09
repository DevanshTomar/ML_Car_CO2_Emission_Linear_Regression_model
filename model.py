from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import aiohttp
import asyncio
import requests
from sklearn import linear_model

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"


# Downloading the file using requests
try:
    response = requests.get(url)
    response.raise_for_status()  
    with open("FuelConsumption.csv", 'wb') as file:
        file.write(response.content)
    print("File downloaded and saved successfully.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")

# Reading the CSV file
df = pd.read_csv("FuelConsumption.csv")

# Selecting relevant columns
df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Displaying the first few rows of the selected columns
print(df.head(9))

# Create a histogram for each selected column
viz = df[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist(bins=30, figsize=(10, 8))
plt.suptitle("Histograms of Selected Features")
plt.show()

# Scatter plot: FUELCONSUMPTION_COMB vs CO2EMISSIONS
plt.figure(figsize=(10, 6))
plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS, color='blue', alpha=0.5)
plt.title("CO2 Emissions vs Fuel Consumption")
plt.xlabel("Fuel Consumption (Combined) [L/100 km]")
plt.ylabel("CO2 Emissions [g/km]")
plt.grid(True)
plt.show()

# Scatter plot: ENGINESIZE vs CO2EMISSIONS
plt.figure(figsize=(10, 6))
plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color='green', alpha=0.5)
plt.title("CO2 Emissions vs Engine Size")
plt.xlabel("Engine Size [L]")
plt.ylabel("CO2 Emissions [g/km]")
plt.grid(True)
plt.show()

# Scatter plot: CYLINDERS vs CO2EMISSIONS
plt.figure(figsize=(10, 6))
plt.scatter(df.CYLINDERS, df.CO2EMISSIONS, color='red', alpha=0.5)
plt.title("CO2 Emissions vs Number of Cylinders")
plt.xlabel("Number of Cylinders")
plt.ylabel("CO2 Emissions [g/km]")
plt.grid(True)
plt.show()

# Spliting the data into training and testing sets
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Scatter plot: Training data (Engine size vs CO2 Emissions)
plt.figure(figsize=(10, 6))
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue', alpha=0.5)
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Engine Size vs CO2 Emissions (Training Data)")
plt.grid(True)
plt.show()

# Creating linear regression model
regr = linear_model.LinearRegression()

# Preparing training data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Training the model
regr.fit(train_x, train_y)

# Printing the coefficients
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

# Ploting the fit line over the training data
plt.figure(figsize=(10, 6))
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue', alpha=0.5)
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Engine Size vs CO2 Emissions with Linear Fit (Training Data)")
plt.grid(True)
plt.show()

# Evaluating the model on the test data
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_pred = regr.predict(test_x)

# Printing evaluation metrics
print("Mean squared error: %.2f" % mean_squared_error(test_y, test_y_pred))
print('R2-score: %.2f' % r2_score(test_y, test_y_pred))

# Scatter plot: Test data (Engine size vs CO2 Emissions)
plt.figure(figsize=(10, 6))
plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='blue', alpha=0.5)
plt.plot(test_x, regr.coef_[0][0] * test_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Engine Size vs CO2 Emissions with Linear Fit (Test Data)")
plt.grid(True)
plt.show()
