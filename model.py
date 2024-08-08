from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import aiohttp
import asyncio
import requests
from sklearn import linear_model

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"


try:
    response = requests.get(url)
    response.raise_for_status()  
    with open("FuelConsumption.csv", 'wb') as file:
        file.write(response.content)
    print("File downloaded and saved successfully.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")


try:
    df = pd.read_csv("FuelConsumption.csv")
    print("File read into DataFrame successfully.")
except Exception as e:
    print(f"Error reading the file into DataFrame: {e}")

# Read the CSV file
df = pd.read_csv("FuelConsumption.csv")

# Select relevant columns
df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Display the first few rows of the selected columns
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
