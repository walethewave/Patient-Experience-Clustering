# Patient Experience Clustering

A data-driven project that leverages patient reviews from Drugs.com to cluster drugs based on effectiveness and user sentiment. This project not only classifies drug effectiveness but also recommends treatments for various conditions based on real-world patient experiences.

---

## Table of Contents

- [Overview](#overview)
- [Purpose](#purpose)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
  - [Effectiveness Classification](#effectiveness-classification)
  - [Clustering with K-Means](#clustering-with-k-means)
  - [Drug & Condition Recommendations](#drug--condition-recommendations)
  - [Visualization](#visualization)
- [Results & Insights](#results--insights)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

**Patient Experience Clustering** is an analytical project that harnesses the power of unsupervised learning and visualization to provide insights into patient reviews of various drugs. By clustering reviews based on ratings and conditions, the project offers actionable recommendations for both patients and healthcare professionals.

---

## Purpose

- **Understand Patient Feedback:** Gain insights into how different drugs perform based on patient reviews.
- **Classify Effectiveness:** Automatically classify drugs as _Effective_, _Moderately Effective_, or _Ineffective_.
- **Cluster Drugs:** Use K-Means clustering to identify natural groupings within the data.
- **Provide Recommendations:** Recommend top-rated drugs for specific conditions based on cluster analysis.
- **Support Decision-Making:** Aid patients and healthcare providers in making informed choices about drug therapies.

---

## Dataset

The dataset is sourced from [Drugs.com](https://www.drugs.com/) and contains the following key columns:
- **drugName:** Name of the drug.
- **condition:** Medical condition treated.
- **review:** Patient review text.
- **rating:** Numeric rating provided by the patient.
- **date:** Date of the review.
- **usefulCount:** Number of users who found the review useful.

Example rows:
| drugName                         | condition                        | review                                          | rating | date       | usefulCount | effectiveness         |
|----------------------------------|----------------------------------|-------------------------------------------------|--------|------------|-------------|-----------------------|
| Valsartan                        | Left Ventricular Dysfunction     | "It has no side effect, I take it in combinati..." | 9.0    | 2012-05-20 | 27          | Effective             |
| Guanfacine                       | ADHD                             | "My son is halfway through his fourth week of ..."| 8.0    | 2010-04-27 | 192         | Effective             |
| Lybrel                           | Birth Control                    | "I used to take another oral contraceptive, wh..." | 5.0    | 2009-12-14 | 17          | Moderately Effective  |
| Ortho Evra                       | Birth Control                    | "This is my first time using any form of birth..." | 8.0    | 2015-11-03 | 10          | Effective             |

---

## Methodology

### Data Cleaning & Preprocessing

- **Handling Missing Data:** Removed rows with missing values in critical columns.
- **Date Conversion:** Converted the date column to a datetime format.
- **Output Cleaning:** Stripped unnecessary output from the notebook to keep file sizes manageable.

### Effectiveness Classification

A custom function classifies drugs based on ratings:
- **Effective:** Rating ≥ 8
- **Moderately Effective:** 5 ≤ Rating < 8
- **Ineffective:** Rating < 5

```python
def classify_effectiveness(rating):
    if rating >= 8:
        return 'Effective'
    elif 5 <= rating < 8:
        return "Moderately Effective"
    else:
        return 'Ineffective'

train_data['effectiveness'] = train_data['rating'].apply(classify_effectiveness)



Below is the content written in Markdown format. You can copy and paste this into your README file or another Markdown document.

---

```markdown
# Clustering with K-Means

**Feature Used:**  
- The numeric `rating` column.

**Normalization:**  
- Standardized ratings using `StandardScaler` to ensure balanced clustering.

**Clustering:**  
- Applied K-Means with an initial choice of **K=5**.  
  > **Note:** On inspection, some clusters remained empty, suggesting that the data may naturally form fewer clusters.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

clustering_features = train_data[['rating']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

# Adjusted K based on data distribution (e.g., K=2 if empty clusters are found)
kmeans = KMeans(n_clusters=5, random_state=42)
train_data['cluster_label'] = kmeans.fit_predict(scaled_features)
```

---

# Drug & Condition Recommendations

Using grouped data, the project computes average ratings per drug and condition within each cluster. A merge operation allows us to recommend the highest-rated drugs for a specific condition and cluster.

```python
# Grouping by cluster and drug
cluster_drug_ratings = train_data.groupby(['cluster_label', 'drugName'])['rating'].mean().reset_index()
top_drugs_per_cluster = cluster_drug_ratings.sort_values(['cluster_label', 'rating'], ascending=[True, False])

# Grouping by cluster and condition
cluster_condition_ratings = train_data.groupby(['cluster_label', 'condition'])['rating'].mean().reset_index()
top_conditions_per_cluster = cluster_condition_ratings.sort_values(['cluster_label', 'rating'], ascending=[True, False])

# Merging and filtering recommendations
recommendations = pd.merge(top_drugs_per_cluster, top_conditions_per_cluster, on='cluster_label', suffixes=('_drug', '_condition'))
recommendations = recommendations[recommendations['rating_drug'] > 8]
```

---
![Screenshot 2025-01-31 200320](https://github.com/user-attachments/assets/6cc6e637-3523-49b1-bfed-5ce7f0d0f9ea)

# Visualization

Visualizations include bar charts and PCA scatter plots to showcase cluster distributions and top recommendations. For instance, the bar chart below displays recommended drugs colored by condition:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.barplot(
    data=recommendations, 
    x='rating_drug', 
    y='drugName', 
    hue='condition', 
    palette='tab10'
)
plt.title('Recommended Drugs by Condition and Cluster')
plt.xlabel('Average Drug Rating')
plt.ylabel('Drug Name')
plt.show()
```

---

# Results & Insights

**Cluster Analysis:**
- **Cluster 0:** Predominantly features highly rated drugs for *Birth Control*, *Depression*, *Pain*, and *Anxiety*.
- **Other Clusters:** Reveal insights into niche drug categories, such as emergency contraception and weight loss treatments.

**Recommendations:**
- The system recommends top-rated drugs within each cluster. For example, in Cluster 0, drugs such as **Drysol**, **Depo-Provera**, and **Chantix** are highly recommended based on patient reviews.

**Visualization:**
- Visual output makes it easy to compare drug performance across various conditions and clusters, helping stakeholders quickly identify effective treatments.

---

# How to Run

## Clone the Repository

```bash
git clone https://github.com/yourusername/Patient-Experience-Clustering.git
cd Patient-Experience-Clustering
```

## Set Up the Environment

Create and activate a virtual environment, then install required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Notebook

Open the Jupyter Notebook:

```bash
jupyter notebook Patient.ipynb
```

## Review Results & Visualizations

- Execute cells to see clustering results, recommendations, and interactive visualizations.
```

