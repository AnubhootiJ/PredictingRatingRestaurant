from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd

# loading the ML model
model = joblib.load("RF_model.pkl")

# loading the dataset
data_file = pd.read_csv("ZomatoDelhi.csv", encoding='latin-1')  # loading the CSV file as the dataframe
data_file = data_file.dropna(axis=0)  # to remove any missing values
features = ['Average_Cost_for_two', 'Has Table booking', 'Has Online delivery', 'Price_range',
            'Votes']  # defining features to be used in model
X = data_file[features]  # the input variables that our model will be fed at the time of training

# fitting the dataset so as to normalize the data point
scaler = MinMaxScaler()
scaler.fit_transform(X)

# Getting values from the user
name = input("Enter name for the restaurant = ")
cost = input("Enter average cost for two = ")
table = input("Enter 1 if it has table booking otherwise 0 = ")
online = input("Enter 1 if it has online delivery otherwise 0 = ")
price = input("Enter price range = ")
vote = input("Average Votes it has = ")

# scaling the data point inputted by the user
d = [[cost, table, online, price, vote]]
# print(d)
scaler.transform(d)

# predicting the rating
answer = model.predict(d)
print("As per the calculations of the model, {} has the predicted rating of {} out of 5.".format(name, answer[0]))
