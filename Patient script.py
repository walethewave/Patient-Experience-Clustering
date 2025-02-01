# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('drugsComTrain_raw.csv')
test_data = pd.read_csv('drugsComTest_raw.csv')
print

train_data.head(5)
train_data.info()


train_data.isnull().sum()
train_data.dropna(inplace=True)
train_data.describe()

train_data.shape
train_data.nunique()


top = train_data['condition'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top[:5].index, y=top[:5].values, color = 'red')
plt.xticks(rotation=45)
plt.title('Top genres', color = 'blue')


top = train_data['drugName'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top[:5].index, y=top[:5].values, color = 'red')
plt.xticks(rotation=45)
plt.title('Top genres', color = 'blue')



train_data = train_data.drop(columns=['Unnamed: 0'])
train_data['date'] = pd.to_datetime(train_data['date'], errors='coerce')
train_data = train_data.dropna(subset=['rating', 'condition', 'drugName'])



#clasifying ratings 

def classify_effectiveness(rating):
    if rating >=8:
        return 'Effective'
    elif 5<= rating <8:
        return "Moderately Effective"
    else:
        return 'Ineffective'
train_data['effectiveness'] = train_data['rating'].apply(classify_effectiveness)



from sklearn.preprocessing import StandardScaler
clustering_features = train_data[['rating']]
                                 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)


from sklearn.cluster import KMeans

KMeans = KMeans(n_clusters =3, random_state=42)
clusters = KMeans.fit_predict(scaled_features)

train_data['cluster_label'] = clusters




print(train_data.groupby('cluster_label')[['drugName', 'condition']].head(6))
# Assuming `kmeans.labels_` contains cluster labels
train_data['cluster_label'] = KMeans.labels_




train_data.groupby('cluster_label')['condition'].value_counts().head(10)
train_data.groupby(['cluster_label', 'condition', 'drugName'])['rating'].mean()


# Check common drugs and conditions per cluster
for cluster in train_data['cluster_label'].unique():
    print(f"\nCluster {cluster}:")
    cluster_data = train_data[train_data['cluster_label'] == cluster]
    print("Top conditions and drugs:")
    print(cluster_data.groupby(['condition', 'drugName']).size().nlargest(5))



# Find the top-rated drugs for each condition in the cluster
top_rated_drugs = (
    train_data.groupby(['cluster_label', 'condition', 'drugName'])['rating']
    .mean()
    .reset_index()
    .sort_values(by=['cluster_label', 'rating'], ascending=False)
)

print(top_rated_drugs.head(10))






def recommend_drugs(cluster_label, condition):
    cluster_data = train_data[
        (train_data['cluster_label'] == cluster_label) & 
        (train_data['condition'] == condition)
    ]
    
    recommended_drugs = (
        cluster_data.groupby('drugName')['rating']
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    return recommended_drugs

# Example usage
print(recommend_drugs(0, "Birth Control"))


#'cluster_label' is the column from clustering
clustered_data = train_data[['drugName', 'condition', 'rating', 'cluster_label']]





# Group data by clusters and drugs
cluster_drug_ratings = clustered_data.groupby(['cluster_label', 'drugName'])['rating'].mean().reset_index()

# Sort within each cluster
top_drugs_per_cluster = cluster_drug_ratings.sort_values(['cluster_label', 'rating'], ascending=[True, False])



# Group data by clusters and conditions
cluster_condition_ratings = clustered_data.groupby(['cluster_label', 'condition'])['rating'].mean().reset_index()

# Sort within each cluster
top_conditions_per_cluster = cluster_condition_ratings.sort_values(['cluster_label', 'rating'], ascending=[True, False])





# Merge the top drugs and conditions by cluster
recommendations = pd.merge(
    top_drugs_per_cluster, 
    top_conditions_per_cluster, 
    on='cluster_label', 
    suffixes=('_drug', '_condition')
)

# Filter for highly rated drugs and conditions
recommendations = recommendations[recommendations['rating_drug'] > 8]




for cluster in recommendations['cluster_label'].unique():
    print(f"Cluster {cluster} Recommendations:")
    cluster_recs = recommendations[recommendations['cluster_label'] == cluster]
    for _, row in cluster_recs.iterrows():
        print(f"  Drug: {row['drugName']} - Condition: {row['condition']} - Drug Rating: {row['rating_drug']:.2f}")





import pandas as pd

# Count how many samples per cluster
cluster_counts = train_data['cluster_label'].value_counts()
print(cluster_counts)




import numpy as np

# Count samples per cluster
cluster_counts = train_data['cluster_label'].value_counts().sort_index()
print(cluster_counts)

# Check if any cluster is missing
missing_clusters = set(range(5)) - set(cluster_counts.index)
print(f"Missing Clusters: {missing_clusters}")




