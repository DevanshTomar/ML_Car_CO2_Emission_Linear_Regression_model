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

print(df.head())

print(df.describe())
