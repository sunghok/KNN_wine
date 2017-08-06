import pandas as pd
import numpy as np
import random
import sklearn.decomposition
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages


data = pd.read_csv("wine.csv")
# print(data.info())
numeric_data = data.drop("color", axis=1)  #color section is redundant. "is_red" exists.




#PCAis a way to take a linear snapshot of the data from several different angles, with each 
#snapshot ordered by how well it aligns with variation in the data.
#standarize the data
numeric_data = (numeric_data - np.mean(numeric_data, axis=0))/np.std(numeric_data, axis=0)

pca = sklearn.decomposition.PCA(n_components=2)
principal_components = pca.fit(numeric_data).transform(numeric_data) 






observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
"""plt.show()"""




#accuracy of the predictions
def accuracy(predictions, outcomes):
    return 100*np.mean(predictions==outcomes)


#check how many wines are "low quality" by using "high,quality"= 0
print(accuracy(0, data["high_quality"]))  #80%


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
print(accuracy(library_predictions, data["high_quality"]))  #99.96%







n_rows = data.shape[0]
random.seed(123)
selection = random.sample(range(n_rows), 10)  #random selection

predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors[selection]])
percentage = accuracy(my_predictions, data.high_quality[selection])
print(percentage)







