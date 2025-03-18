import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("house-prices.csv")
df = pd.DataFrame(data)

X = df[["SqFt", "Bedrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

model = LinearRegression()
model.fit(X_train, y_train)

with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

try:
    with open("history.pkl", "rb") as file:
        history = pickle.load(file)
    if not isinstance(history, list):  
        history = []  
except (FileNotFoundError, pickle.UnpicklingError):  
    history = []

with open("history.pkl", "wb") as file:
    pickle.dump(history, file)
