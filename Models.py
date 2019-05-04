#  import all the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  # import MAE method from sklearn.metrics

data_file = pd.read_csv("ZomatoDelhi.csv", encoding='latin-1')  # loading the CSV file as the dataframe
data_file = data_file.dropna(axis=0)  # to remove any missing values
data_file.describe()  # only works with numeric values


y = data_file.Aggregate_rating  # target column, one our model will predict - the output variable
features = ['Average_Cost_for_two', 'Has Table booking', 'Has Online delivery', 'Price_range',
            'Votes']  # defining features to be used in model
X = data_file[features]  # the input variables that our model will be fed at the time of training

#  splitting the model into training and testing set
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)


#  ###########  --- SUPPORT VECTOR REGRESSION --- ########### #
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # training the model using SVR
svr_rbf.fit(train_X, train_y)  # Fit model
y_rbf = svr_rbf.predict(test_X)  # predicting the values of testing set
result = [test_y.values, y_rbf]  # getting the results of SVR prediction
res = pd.DataFrame(result, index=['test_y', 'y_rbf']).transpose()
print(res.head())


#  ###########  --- DATA NORMALIZATION --- ########### #
min_val = train_X.min()
range_train =  (train_X - min_val).max()
train_X_scaled = (train_X - min_val)/range_train

min_test = test_X.min()
range_test =  (test_X - min_test).max()
test_X_scaled = (test_X - min_test)/range_test
# print(train_X)


#  ###########  --- SVR after normalization --- ########### #
svr_rbf.fit(train_X_scaled, train_y)
y_rbf_scaled = svr_rbf.predict(test_X_scaled)
result = [test_y.values, y_rbf_scaled]  # getting the results of SVR prediction after normalization
res = pd.DataFrame(result, index=['test_y', 'y_rbf_scaled']).transpose()
print(res.head())


#  ###########  --- SVR with optimized values for C and gamma --- ########### #
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=4)
grid.fit(train_X_scaled, train_y)
# grid.best_params_
grid_predictions = grid.predict(test_X_scaled)
result = [test_y.values, grid_predictions]  # getting the results of SVR prediction using optimized values for C and gamma
res = pd.DataFrame(result, index=['test_y', 'grid_predictions']).transpose()
print(res.head())


#  ###########  --- LINEAR REGRESSION --- ########### #
linearRegressor = LinearRegression()
linearRegressor.fit(train_X_scaled, train_y)
y_predict = linearRegressor.predict(test_X_scaled)
result = [test_y.values, y_predict]  # getting the results of Linear Regression
res = pd.DataFrame(result, index=['test_y', 'y_predict']).transpose()
print(res.head())


#  ###########  --- LASSO REGRESSION --- ########### #
reg = LassoCV(cv=10, random_state=0)
reg.fit(train_X_scaled, train_y)
y_lasso = reg.predict(test_X_scaled)
result = [test_y.values, y_lasso]  # getting the results of Lasso Regression
res = pd.DataFrame(result, index=['test_y', 'y_lasso']).transpose()
print(res.head())


#  ###########  --- DECISION TREE REGRESSION --- ########### #
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(train_X_scaled, train_y)
y_tree = tree.predict(test_X_scaled)
result = [test_y.values, y_tree]  # getting the results of Decision Tree Regression
res = pd.DataFrame(result, index=['test_y', 'y_tree']).transpose()
print(res.head())


#  ###########  --- RANDOM FOREST REGRESSION --- ########### #
reg2 = RandomForestRegressor(n_estimators=100, max_depth =3, random_state=0)
reg2.fit(train_X_scaled, train_y)
y_forest = reg2.predict(test_X_scaled)
result = [test_y.values, y_forest]  # getting the results of Random Forest Regression
res = pd.DataFrame(result, index=['test_y', 'y_forest']).transpose()
print(res.head())


#  seeing the results
result = [test_y.values, y_rbf, y_rbf_scaled, grid_predictions, y_predict, y_lasso, y_tree, y_forest]
res = pd.DataFrame(result, index=['Target Values', 'SVR', 'SVR scaled data', 'SVR Optimized', 'Linear Reg', 'Lasso', 'Tree', 'Forest']).transpose()
print(res.head())


#  MAE Values for all the models
e1 = mean_absolute_error(test_y, y_rbf)  # MAE for SVR
print("MAE for SVR = ", e1)
e2 = mean_absolute_error(test_y, y_rbf_scaled)  # MAE for SVR after data normalization
print("MAE for SVR after data normalization = ", e2)
e3 = mean_absolute_error(test_y, grid_predictions)  # MAE for SVR after data normalization with optimized C and gamma
print("MAE for SVR with optimized C and gamma = ", e3)
e4 = mean_absolute_error(test_y, y_predict)  # MAE for Linear Regression
print("MAE for Linear Regression = ", e4)
e5 = mean_absolute_error(test_y, y_lasso)  # MAE for Lasso Regression
print("MAE for Lasso Regression = ", e5)
e6 = mean_absolute_error(test_y, y_tree)  # MAE for Decision Tree Regression
print("MAE for Decision Tree Regression = ", e6)
e7 = mean_absolute_error(test_y, y_forest)  # MAE for Random Forest Regression
print("MAE for Random Forest Regression = ", e7)
