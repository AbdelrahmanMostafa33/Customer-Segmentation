# ==========================================================
# 🛍️ Customer Segmentation using K-Means and DBSCAN
# ==========================================================
# ========== 1. Import Libraries ==========
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import Counter
import io
# For reproducibility
RANDOM_STATE = 0

# ========== 2. Load Dataset ==========
csv_data = """CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Female,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14
26,Male,29,28,82
27,Female,45,28,32
28,Male,35,28,61
29,Female,40,29,31
30,Female,23,29,87
31,Male,60,30,4
32,Female,21,30,73
33,Male,53,33,4
34,Male,18,33,92
35,Female,49,33,14
36,Female,21,33,81
37,Female,42,34,17
38,Female,30,34,73
39,Female,36,37,26
40,Female,20,37,75
41,Female,65,38,35
42,Male,24,38,92
43,Male,48,39,36
44,Female,31,39,61
45,Female,49,39,28
46,Female,24,39,65
47,Female,50,40,55
48,Female,27,40,47
49,Female,29,40,42
50,Female,31,40,42
51,Female,49,42,52
52,Male,33,42,60
53,Female,31,43,54
54,Male,59,43,60
55,Female,50,43,45
56,Male,47,43,41
57,Female,51,44,50
58,Male,69,44,46
59,Female,27,46,51
60,Male,53,46,46
61,Male,70,46,56
62,Male,19,46,55
63,Female,67,47,52
64,Female,54,47,59
65,Male,63,48,51
66,Male,18,48,59
67,Female,43,48,50
68,Female,68,48,48
69,Male,19,48,59
70,Female,32,48,47
71,Male,70,49,55
72,Female,47,49,42
73,Female,60,50,49
74,Female,60,50,56
75,Male,59,54,47
76,Male,26,54,54
77,Female,45,54,53
78,Male,40,54,48
79,Female,23,54,52
80,Female,49,54,42
81,Male,57,54,51
82,Male,38,54,55
83,Male,67,54,41
84,Female,46,54,44
85,Female,21,54,57
86,Male,48,54,46
87,Female,55,57,58
88,Female,22,57,55
89,Female,34,58,60
90,Female,50,58,46
91,Female,68,59,55
92,Male,18,59,41
93,Male,48,60,49
94,Female,40,60,40
95,Female,32,60,42
96,Male,24,60,52
97,Female,47,60,47
98,Female,27,60,50
99,Male,48,61,42
100,Male,20,61,49
101,Female,23,62,41
102,Female,49,62,48
103,Male,67,62,59
104,Male,26,62,55
105,Male,49,62,56
106,Female,21,62,42
107,Female,66,63,50
108,Male,54,63,46
109,Male,68,63,43
110,Male,66,63,48
111,Male,65,63,52
112,Female,19,63,54
113,Female,38,64,42
114,Male,19,64,46
115,Female,18,65,48
116,Female,19,65,50
117,Female,63,65,43
118,Female,49,65,59
119,Female,51,67,43
120,Female,50,67,57
121,Male,27,67,56
122,Female,38,67,40
123,Female,40,69,58
124,Male,39,69,91
125,Female,23,70,29
126,Female,31,70,77
127,Male,43,71,35
128,Male,40,71,95
129,Male,59,71,11
130,Male,38,71,75
131,Male,47,71,9
132,Male,39,71,75
133,Female,25,72,34
134,Female,31,72,71
135,Male,20,73,5
136,Female,29,73,88
137,Female,44,73,7
138,Male,32,73,73
139,Male,19,74,10
140,Female,35,74,72
141,Female,57,75,5
142,Male,32,75,93
143,Female,28,76,40
144,Female,32,76,87
145,Male,25,77,12
146,Male,28,77,97
147,Male,48,77,36
148,Female,32,77,74
149,Female,34,78,22
150,Male,34,78,90
151,Male,43,78,17
152,Male,39,78,88
153,Female,44,78,20
154,Female,38,78,76
155,Female,47,78,16
156,Female,27,78,89
157,Male,37,78,1
158,Female,30,78,78
159,Male,34,78,1
160,Female,30,78,73
161,Female,56,79,35
162,Female,29,79,83
163,Male,19,81,5
164,Female,31,81,93
165,Male,50,85,26
166,Female,36,85,75
167,Male,42,86,20
168,Female,33,86,95
169,Female,36,87,27
170,Male,32,87,63
171,Male,40,87,13
172,Male,28,87,75
173,Male,36,87,10
174,Male,36,87,92
175,Female,52,88,13
176,Female,30,88,86
177,Male,58,88,15
178,Male,27,88,69
179,Male,59,93,14
180,Male,35,93,90
181,Female,37,97,32
182,Female,32,97,86
183,Male,46,98,15
184,Female,29,98,88
185,Female,41,99,39
186,Male,30,99,97
187,Female,54,101,24
188,Male,28,101,68
189,Female,41,103,17
190,Female,36,103,85
191,Female,34,103,23
192,Female,32,103,69
193,Male,33,113,8
194,Female,38,113,91
195,Female,47,120,16
196,Female,35,120,79
197,Female,45,126,28
198,Male,32,126,74
199,Male,32,137,18
200,Male,30,137,83"""
df = pd.read_csv(io.StringIO(csv_data))

# ========== 3. Basic Information ==========
print("Dataset Preview:")
print(df.head(), "\n")
print('Shape:', df.shape)
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# ========== 4. Data Cleaning ==========
# 1. Check duplicates
print('Duplicate rows:', df.duplicated().sum())
# 2. Check missing values
print('\nMissing values per column:')
print(df.isna().sum())
# 3. Ensure column names are consistent (trim spaces)
df.columns = [c.strip() for c in df.columns]
# Drop duplicates if any
if df.duplicated().sum() > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print('Dropped duplicates. New shape:', df.shape)

# ========== 5. Exploratory Data Analysis (EDA) ==========
# Histogram + KDE
num_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
for ax, col in zip(axes, num_cols):
    sns.histplot(df[col].dropna(), kde=True, bins=20, color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
plt.suptitle("Histogram + KDE Distributions", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

# Boxplots
fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col].dropna(), color='skyblue', ax=ax)
    ax.set_title(col)
    ax.set_ylabel('Value')
    ax.set_xlabel('')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.suptitle("Boxplots of Numerical Features", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

# Scatter plot: Annual Income vs Spending Score
x_col, y_col = 'Annual Income (k$)', 'Spending Score (1-100)'
plt.figure(figsize=(6, 4))
plt.scatter(df[x_col], df[y_col], s=30, color='skyblue', edgecolor='black')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f'{x_col} vs {y_col}')
plt.tight_layout()
plt.show()

# ==========================================================
# 🧩 K-MEANS CLUSTERING
# ==========================================================
# ========== 6. Feature Selection and Scaling ==========
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 7. Find Optimal k (Elbow Method and Silhouette Scores) ==========
inertias = []
sil_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_)
    sil_scores.append(sil)

# Plot inertia
plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(K)
plt.show()

# Plot silhouette
plt.figure(figsize=(6,4))
plt.plot(list(K), sil_scores, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for different k')
plt.xticks(K)
plt.show()

# Optimal k = 5 based on analysis

# ========== 8. Fit KMeans with Optimal k ==========
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels
sil_score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for k={optimal_k}: {sil_score:.3f}\n")

# Quick cluster counts
print('Cluster counts:')
print(df['Cluster'].value_counts().sort_index())

# Cluster profile (after dropping non-feature columns)
df_temp = df.drop(columns=['CustomerID', 'Gender', 'Age'])
cluster_profile = df_temp.groupby('Cluster').mean()
print('\nCluster profile (averages):')
print(cluster_profile)

# ========== 9. Average Spending per Cluster ==========
avg_spending = df_temp.groupby('Cluster')['Spending Score (1-100)'].mean().sort_index()
avg_spending.plot(kind='bar', figsize=(6,4))
plt.title('Average Spending Score per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending Score')
plt.show()

# ========== 10. Visualize KMeans Clusters ==========
palette = sns.color_palette("Set1", n_colors=kmeans.n_clusters)
plt.figure(figsize=(8,6))
# Plot points colored by cluster
for i in range(kmeans.n_clusters):
    plt.scatter(
        X.loc[df['Cluster'] == i, 'Annual Income (k$)'],
        X.loc[df['Cluster'] == i, 'Spending Score (1-100)'],
        s=40,
        c=[palette[i]],
        edgecolor='black',
        alpha=0.7,
        label=f'Cluster {i}'
    )
# Centroids (in original scale)
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_orig[:,0],
    centroids_orig[:,1],
    marker='X',
    s=250,
    c='yellow',
    edgecolor='black',
    linewidth=1.2,
    label='Centroids'
)
plt.xlabel('Annual Income (k$)', fontsize=11)
plt.ylabel('Spending Score (1-100)', fontsize=11)
plt.title(f'KMeans Clusters (k={kmeans.n_clusters})', fontsize=14, pad=10)
plt.legend(title='Clusters', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================
# 🔍 DBSCAN CLUSTERING
# ==========================================================
# ========== 11. Apply DBSCAN ==========
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = db_labels
print("DBSCAN Cluster Counts (including -1 as noise):")
print(Counter(db_labels), "\n")

# ========== 12. Visualize DBSCAN Clusters ==========
unique_labels = sorted(set(db_labels))
palette = sns.color_palette("Set2", n_colors=len(unique_labels))
plt.figure(figsize=(8,6))
for label in unique_labels:
    color = 'k' if label == -1 else palette[label % len(palette)]
    plt.scatter(
        X.loc[df['DBSCAN_Cluster'] == label, 'Annual Income (k$)'],
        X.loc[df['DBSCAN_Cluster'] == label, 'Spending Score (1-100)'],
        s=40,
        c=[color],
        edgecolor='black',
        alpha=0.7,
        label='Noise' if label == -1 else f'Cluster {label}'
    )
plt.xlabel('Annual Income (k$)', fontsize=11)
plt.ylabel('Spending Score (1-100)', fontsize=11)
plt.title('DBSCAN Clustering (eps=0.5, min_samples=5)', fontsize=14, pad=10)
plt.legend(title='Clusters', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================
# 🏁 Save Segmented Dataset
# ==========================================================
output_path = "../data/processed/Mall_Customers_Segmented.csv"
df.to_csv(output_path, index=False)
print(f"✅ Segmented dataset saved successfully to {output_path}")

# ==========================================================
# 📊 Final Summary
# ==========================================================
print("""
🏁 Conclusion
In this analysis, we segmented mall customers based on their Annual
Income and Spending Score to uncover distinct behavioral groups.
After scaling and applying the Elbow and Silhouette methods, five
clusters (k = 5) were selected as the most suitable structure for the
data.
The K-Means clustering revealed clear customer segments:
- Cluster 0: Middle-income, average spenders — a balanced group.
- Cluster 1: High-income, high-spending customers — premium target
    group.
- Cluster 2: High-income, low-spending customers — potential for
    marketing focus to increase engagement.
- Cluster 3: Low-income, low-spending customers — least profitable
    segment.
- Cluster 4: Low-income, high-spending customers — enthusiastic
    spenders despite limited income.
The average spending score visualization confirmed strong differences
between clusters, validating the segmentation.
Additionally, DBSCAN was tested as an alternative density-based
algorithm. While it identified two main clusters and a few outliers,
K-Means provided more interpretable and balanced groupings for this
dataset.
Overall, the analysis successfully demonstrates how unsupervised
learning can uncover actionable customer segments that can guide
targeted marketing strategies and personalized offers.
""")
print("Script execution completed successfully.")