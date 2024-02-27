import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv(r'/Users/mohammad/Downloads/Uni/Term 8/bank-sample-test.csv')

X = data.loc[:, data.columns != 'subscribed']
Y = data.loc[:, data.columns == 'subscribed']

x = X.to_numpy()
y = Y.to_numpy().ravel()


## DBscan Algo ###
neighb = NearestNeighbors(n_neighbors=5) # creating an object of the NearestNeighbors class
nbrs = neighb.fit(x) # fitting the data to the object
distances,indices = nbrs.kneighbors(x) # finding the nearest neighbours

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
plt.plot(distances) # plotting the distances
plt.show() # showing the plot

dbscan = DBSCAN(eps = 400, min_samples = 4).fit(x) # fitting the model
labels = dbscan.labels_ # getting the labels

# Plot the clusters
plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma") # plotting the clusters
plt.xlabel("Income") # X-axis label
plt.ylabel("Spending Score") # Y-axis label
plt.show() # showing the plot
