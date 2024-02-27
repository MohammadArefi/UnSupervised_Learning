import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
data = pd.read_csv(r'/Users/mohammad/Downloads/Uni/Term 8/bank-sample-test.csv')
X = data.loc[:, data.columns != 'subscribed']
Y = data.loc[:, data.columns == 'subscribed']
x = X.to_numpy()
y = Y.to_numpy().ravel()

silhouette_score_1 = round(silhouette_score(x, y), 1)
print('Silhouette Score 1:', silhouette_score_1)

silhouette_score_2 = round(silhouette_score(x, y), 2)
print('Silhouette Score 2:', silhouette_score_2)

silhouette_score_3 = round(silhouette_score(x, y), 3)
print('Silhouette Score 3:', silhouette_score_3)

silhouette_score_4 = round(silhouette_score(x, y), 4)
print('Silhouette Score 4:', silhouette_score_4)

silhouette_score_5 = round(silhouette_score(x, y), 5)
print('Silhouette Score 5:', silhouette_score_5)

silhouette_score_6 = round(silhouette_score(x, y), 6)
print('Silhouette Score 6:', silhouette_score_6)

silhouette_score_7 = round(silhouette_score(x, y), 7)
print('Silhouette Score 7:', silhouette_score_7)

# Creating the BIRCH clustering model
model = Birch(branching_factor = 50, n_clusters = 5, threshold = 1.5)
# Fit the data (Training)
birch = model.fit(data)

# Predict the same data
pred = model.predict(data)
silhouette_score_birch = round(silhouette_score(x, model.labels_, metric='euclidean'))
print('Silhouette Score birch:', silhouette_score_birch)

# Creating a scatter plot
plt.scatter(x[:,0], x[:,1], c = pred, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')
plt.show()
