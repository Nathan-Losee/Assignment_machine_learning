import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
pd.set_option('display.max_columns', 1000)
# Load the dataset
column_names = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
data = pd.read_csv('breast-cancer.data', header=None, names=column_names)
data['tumor-size'] = data['tumor-size'].apply(lambda x: '05-09' if x == '5-9' else x)
print(data.head())

label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])
# Dropping rows with missing values
data = data.dropna()
print(data.head())
# Selecting features for clustering
X = data.drop('Class', axis=1)

# Calculate SSE for different values of k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
print('Sum of Square Errors = ', sse)

# Evaluate silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot silhouette scores for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()

# Plot SSE for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')


optimal_k = 7  # Elbow method showed that around 7 is when the curve starts to even out
plt.plot(optimal_k, sse[optimal_k - 1], marker='o', markersize=8, color='red')
plt.annotate(f'Optimal k={optimal_k}', (optimal_k, sse[optimal_k - 1]), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=10)
plt.show()

print("Considering the Elbow method along with the silhouette method, I believe the best number of clusters to be 7, as it really tapers off after 7 clusters.")
# Fit KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

# Plotting clusters and centroids (using two features for visualization)
feature1 = 'age'  # Replace with one of the features from the dataset
feature2 = 'tumor-size'  # Replace with another feature from the dataset

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[feature1], X[feature2], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, X.columns.get_loc(feature1)], kmeans.cluster_centers_[:, X.columns.get_loc(feature2)], c='red', marker='o', edgecolors='black', s=200, label='Centroids')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'KMeans Clustering with {optimal_k} Clusters')

# Annotate centroids with cluster numbers
for i, centroid in enumerate(kmeans.cluster_centers_):
    plt.text(centroid[X.columns.get_loc(feature1)], centroid[X.columns.get_loc(feature2)], str(i+1), fontsize=12, color='black', ha='center', va='center')


age_ticks = sorted(data[feature1].unique())
age_labels = {
    0: '20-29',
    1: '30-39',
    2: '40-49',
    3: '50-59',
    4: '60-69',
    5: '70-79'
}
tumor_ticks = sorted(data[feature2].unique())
tumor_labels = {
    0: '0-4',
    1: '05-09',
    2: '10-14',
    3: '15-19',
    4: '20-24',
    5: '25-29',
    6: '30-34',
    7: '35-39',
    8: '40-44',
    9: '45-49',
    10: '50-54'
}
plt.xticks(age_ticks, [age_labels[t] for t in age_ticks])
plt.yticks(tumor_ticks, [tumor_labels[t] for t in tumor_ticks])

plt.legend()
plt.show()


print('This isnt that great when we consider a lot of this data is catagorical, and not a lot of numerical data. If this dataset contained more numerical, I believe we would have better clusters and better looking data that what we have.')
print('Sum of Square Errors in case you missed it earlier = ', sse)