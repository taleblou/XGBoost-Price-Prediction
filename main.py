import xgboost as xgb



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from keras import Sequential
from keras.src.layers import GRU, Dropout, Dense
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, \
    explained_variance_score
import yfinance as yf
import os
from tqdm import tqdm
import warnings
import tensorflow as tf
print(tf.__version__)
warnings.filterwarnings("ignore")
print("GPUs:", tf.config.list_physical_devices('GPU'))
# Check if GPU is available
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Using device: {device}")

name="BTC-USD"#"GC=F""EURUSD=X""^GSPC"
file_path = f"xgboost{name}.txt"
def text_write(text):
    print(text)
    # Check if the file exists
    if os.path.exists(file_path):
        # Open the file in append mode ('a') to add text without overwriting
        with open(file_path, 'a') as file:
            file.write(text+"\n")
    else:
        # If the file doesn't exist, create it and write the first line
        with open(file_path, 'w') as file:
            file.write(text+"\n")

# 1. Load and preprocess the data
def load_data(ticker):
    # Load data from Yahoo Finance
    data = yf.download(ticker)
    return data



def build_model(X_train,Y_train):

    # Define the model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42)

    # Hyperparameter tuning using GridSearchCV
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'subsample': [0.7, 0.8, 1.0]
    # }
    # grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error',
    #                            verbose=2)
    # grid_search.fit(X_train, Y_train)
    #
    # # Best model
    # best_model = grid_search.best_estimator_
    # print("Best parameters found: ", grid_search.best_params_)
    xgb_model.fit(X_train, Y_train)
    return xgb_model

data = load_data(name)
data.to_csv(f"{name}.csv")
# data = load_data("GC=F")
# data.to_csv("GC=F.csv")

data=data[["Close","Open","High","Low"]]
data=data[-1000:]
# Initialize the scaler
scaler = MinMaxScaler()
# Fit and transform the DataFrame
scaled_data = scaler.fit_transform(data)
# Convert the result back to a DataFrame with the same column names
data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)





data["y_Close"]=data['Close']
data["y_Close"]=data["y_Close"].shift(-1)
data["y_Open"]=data['Open']
data["y_Open"]=data["y_Open"].shift(-1)
data["y_High"]=data['High']
data["y_High"]=data["y_High"].shift(-1)
data["y_Low"]=data['Low']
data["y_Low"]=data["y_Low"].shift(-1)
data.dropna(inplace=True)

X=data[["Close","Open","High","Low"]]
Y=data[["y_Close","y_Open","y_High","y_Low"]]
# Train-test split
data["p_Low"]= np.nan
data["p_High"]= np.nan
data["p_Open"]= np.nan
data["p_Close"]= np.nan


data["o_p_Low"]= np.nan
data["o_p_High"]= np.nan
data["o_p_Open"]= np.nan
data["o_p_Close"]= np.nan

data["o_y_Low"]= np.nan
data["o_y_High"]= np.nan
data["o_y_Open"]= np.nan
data["o_y_Close"]= np.nan






box=200
for i in tqdm(range(box-1)):
    X_train=X[:i-box]
    Y_train=Y[:i-box]
    X_test = X[i-box:i - box+1]
    Y_test = Y[i-box:i - box+1]
    for c in ['Open','High','Low','Close']:
        Xtrain=X_train[[c]]
        Xtrain = np.array(Xtrain)  # Add timestep dimension (samples, timesteps, features)

        Xtest = np.expand_dims(X_test, axis=1)

        model=build_model(Xtrain, Y_train["y_"+c])

        predictions = model.predict(Y_test["y_" + c].values.reshape(-1, 1))
        predictions=np.array(predictions)
        data.loc[data.index[i - box + 1], 'p_'+c]=predictions[0]
        predictions=np.tile(predictions, 4).reshape(1, 4)
        predictions = scaler.inverse_transform(predictions)
        data.loc[data.index[i - box + 1], 'o_p_'+c] = predictions[0][0]
        target = scaler.inverse_transform(Y_test)
        data.loc[data.index[i - box + 1], 'o_y_'+c] = target[0][0]



# Calculate Accuracy (for classification)
df= data[['y_Open','p_Open','y_Close','p_Close','y_High','p_High','y_Low','p_Low','o_y_Open','o_p_Open','o_y_Close','o_p_Close','o_y_High','o_p_High','o_y_Low','o_p_Low']]
df.dropna(inplace=True)
df.to_csv(f"Predict_{name}.csv")

# Plot separate line charts
for c in ['Open','High','Low','Close']:
    # Mean Squared Error (MSE)
    mse = mean_squared_error(df['y_' + c], df['p_' + c])

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(df['y_' + c], df['p_' + c])

    # R-squared (R2)
    r2 = r2_score(df['y_' + c], df['p_' + c])

    # Median Absolute Error
    medae = median_absolute_error(df['y_' + c], df['p_' + c])

    #Explained Variance Score
    evs = explained_variance_score(df['y_' + c], df['p_' + c])

    text_write(f"Mean Squared Error({c}): {mse}")
    text_write(f"Mean Absolute Error({c}): {mae}")
    text_write(f"R-squared({c}): {r2}")
    text_write(f"Median Absolute Error({c}): {medae}")
    text_write(f"Explained Variance Score({c}): {evs}")
    fig, axes = plt.subplots()

    # Open price plot
    plt.plot(df.index, df['o_y_'+c], label='Actual '+c+' Price', color='blue')
    plt.plot(df['o_p_'+c], label='Predicted '+c+' Price', color='green')
    plt.title(f'{c} Price Prediction')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"xgboost_{c}_{name}.png")
    plt.close()
    plt.cla()


