#  import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_file = pd.read_csv("ZomatoDelhi.csv", encoding='latin-1')  # loading the CSV file as the dataframe
data_file = data_file.dropna(axis=0)  # to remove any missing values
data_file.describe()  # only works with numeric values

print(data_file.columns.values)  # to get all the columns

features = ['Average_Cost_for_two', 'Has Table booking', 'Has Online delivery', 'Price_range',
            'Votes']  # defining features to be used in model
X = data_file[features]  # the input variables that our model will be fed at the time of training

ag = data_file["Aggregate_rating"]
vo = data_file["Votes"]
av = data_file["Average_Cost_for_two"]

#  relation between the number of votes given by the customer to a particular restaurant and the rating given to that restaurant.
plt.figure(figsize=[6, 6])
plt.scatter(ag, vo)
plt.xlabel("Aggregate Rating")
plt.ylabel("Votes")
plt.savefig("Visual1.png")

#  relation between Average Cost for two people and the rating given to that restaurant.

plt.figure(figsize=[6, 6])
plt.scatter(ag, av)
plt.xlabel("Aggregate Rating")
plt.ylabel("Average cost for two")
plt.savefig("Visual2.png")

#  heatmap to see relation between variables
ax = sns.heatmap(X.corr(), annot=True)
fig = ax.get_figure()
fig.savefig("Visual3.png")
fig.clf()

#  count of restaurants as per the ratings (in text format) they received
v4 = sns.countplot(data_file['Rating text'])
fig1 = v4.get_figure()
fig1.savefig("Visual4.png")
fig1.clf()

#  Count number of restaurants based on price range in three cities – Gurgaon, New Delhi, and Noida.
v5 = sns.countplot(data_file['City'], hue=data_file['Price_range'])
fig2 = v5.get_figure()
fig2.savefig("Visual5.png")
fig2.clf()

#  number of restaurants based on price range and whether the restaurant provides online delivery or not
v6 = sns.countplot(data_file['Price_range'], hue=data_file['Has Online delivery'])
fig3 = v6.get_figure()
fig3.savefig("Visual6.png")
fig3.clf()

#  number of restaurants based on price range and whether they provide table booking or not.
v7 = sns.countplot(data_file['Price_range'], hue=data_file['Has Table booking'])
fig4 = v7.get_figure()
fig4.savefig("Visual7.png")
fig4.clf()

#  Violin plot for Aggregate Rating Vs Price Range
v8 = sns.violinplot(data_file['Price_range'], data_file['Aggregate_rating'])
fig5 = v8.get_figure()
fig5.savefig("Visual8.png")
fig5.clf()

#  Violin plot for Aggregate Rating Vs Table Booking
v9 = sns.violinplot(data_file['Has Table booking'], data_file['Aggregate_rating'])
fig6 = v9.get_figure()
fig6.savefig("Visual9.png")
fig6.clf()

#  Violin plot for Aggregate Rating Vs Online delivery
v10 = sns.violinplot(data_file['Has Online delivery'], data_file['Aggregate_rating'])
fig7 = v10.get_figure()
fig7.savefig("Visual10.png")
fig7.clf()
