from fastapi import FastAPI
from fastapi.responses import FileResponse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import uuid
import os

app = FastAPI()

df = sns.load_dataset("CCPP_data.csv")   # or your CSV

# 1. Generate pairplot
@app.get("/pairplot")
def pairplot():
    import seaborn as sns

    file = f"/tmp/{uuid.uuid4()}.png"
    sns.pairplot(df).savefig(file)
    return FileResponse(file, media_type="image/png")


# 2. Model training endpoint
@app.get("/model")
def model(type: str = "lr"):
    X = df[["sepal_length", "sepal_width"]]
    y = df["petal_length"]

    if type == "lr":
        model = LinearRegression()
    elif type == "rf":
        model = RandomForestRegressor()
    else:
        return {"error": "unknown model"}

    model.fit(X, y)
    pred = model.predict(X)

    file = f"/tmp/{uuid.uuid4()}.png"
    plt.figure()
    plt.scatter(y, pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{type.upper()} results")
    plt.savefig(file)

    return FileResponse(file, media_type="image/png")
