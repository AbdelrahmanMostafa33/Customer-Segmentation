# main.py
# Customer Segmentation

# Data handling
import pandas as pd
import numpy as np
import math

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning (Scikit-learn)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

def main():
    print("Customer Segmentation Analysis")
    print("=" * 50)
    
    # 2️⃣ Data Loading & Cleaning
    print("\n## 2️⃣ Data Loading & Cleaning")
    
    # Load the dataset and quick overview
    df = pd.read_csv("../data/Mall_Customers.csv")
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    df.info()
    
    print("\nDataset description:")
    print(df.describe())
    
    # Check for duplicates
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isna().sum())
    
    # Ensure correct data types
    print("\nData types:")
    print(df.dtypes)
    
    # 3️⃣ Exploratory Data Analysis (EDA)
    print("\n## 3️⃣ Exploratory Data Analysis (EDA)")
    
    # Univariate Analysis
    numerical_cols = df.select_dtypes(include=['int64']).columns.drop('CustomerID').to_list()
    
    # Automatically calculate rows and columns
    n_cols = 3
    n_rows = math.ceil(len(numerical_cols) / n_cols)
    
    # Create histograms for numerical features
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, bins=20, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    # Create boxplots for numerical features
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)
    
    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
    
    # Bivariate Analysis: Relationship between Annual Income and Spending Score
    print("\n## Bivariate Analysis: Income vs Spending Score")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
    plt.title('Annual Income vs Spending Score')
    plt.show()
    
    # Correlation heatmap
    print("\n## Correlation Heatmap")
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()
    
    # 4️⃣ Clustering Analysis
    print("\n## 4️⃣ Clustering Analysis")
    
    # Prepare data for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using Elbow Method
    print("\n## Elbow Method for Optimal Clusters")
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    cluster_range = range(2, 11)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print(f"Clusters: {k}, WCSS: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette_avg:.3f}")
    
    # Plot Elbow Method
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(cluster_range, wcss, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(cluster_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Choose optimal number of clusters (based on elbow and silhouette)
    optimal_clusters = 5  # Based on typical mall customer segmentation
    
    # Apply KMeans with optimal clusters
    print(f"\n## Applying KMeans with {optimal_clusters} clusters")
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to dataframe
    df['Cluster'] = kmeans.labels_
    
    # Visualize clusters
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='red', marker='X', label='Centroids')
    plt.xlabel('Annual Income (Standardized)')
    plt.ylabel('Spending Score (Standardized)')
    plt.title('KMeans Clustering (Standardized Data)')
    plt.legend()
    plt.colorbar(scatter)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                         c=kmeans.labels_, cmap='viridis')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('KMeans Clustering (Original Data)')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze clusters
    print("\n## Cluster Analysis")
    cluster_summary = df.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std'],
        'CustomerID': 'count'
    }).round(2)
    
    print("Cluster Summary:")
    print(cluster_summary)
    
    # 5️⃣ Advanced Analysis (Bonus)
    print("\n## 5️⃣ Advanced Analysis (Bonus)")
    
    # DBSCAN Clustering
    print("\n## DBSCAN Clustering")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Count unique clusters (excluding noise points labeled as -1)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN - Estimated number of clusters: {n_clusters_dbscan}")
    print(f"DBSCAN - Estimated number of noise points: {n_noise}")
    
    # Visualize DBSCAN results
    plt.figure(figsize=(8, 6))
    unique_labels = set(dbscan_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = [0, 0, 0, 1]
        
        class_member_mask = (dbscan_labels == k)
        xy = X_scaled[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.7)
    
    plt.xlabel('Annual Income (Standardized)')
    plt.ylabel('Spending Score (Standardized)')
    plt.title(f'DBSCAN Clustering\nEstimated clusters: {n_clusters_dbscan}')
    plt.show()
    
    # Average spending per cluster
    print("\n## Average Spending per Cluster")
    avg_spending = df.groupby('Cluster')['Spending Score (1-100)'].mean().sort_values(ascending=False)
    print("Average Spending Score by Cluster:")
    for cluster, spending in avg_spending.items():
        print(f"Cluster {cluster}: {spending:.2f}")
    
    # Cluster interpretation
    print("\n## Cluster Interpretation")
    cluster_interpretation = {
        0: "High Income, Low Spending - Careful Customers",
        1: "Medium Income, Medium Spending - Standard Customers", 
        2: "High Income, High Spending - Target Customers",
        3: "Low Income, High Spending - Careless Customers",
        4: "Low Income, Low Spending - Sensible Customers"
    }
    
    for cluster_num in range(optimal_clusters):
        cluster_data = df[df['Cluster'] == cluster_num]
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        
        print(f"\nCluster {cluster_num}:")
        print(f"  Interpretation: {cluster_interpretation.get(cluster_num, 'Unknown')}")
        print(f"  Average Income: ${avg_income:.2f}k")
        print(f"  Average Spending Score: {avg_spending:.2f}")
        print(f"  Number of Customers: {len(cluster_data)}")
    
    # 6️⃣ Summary and Insights
    print("\n## 6️⃣ Summary and Insights")
    print("\nKey Findings:")
    print("1. Customers can be segmented into 5 distinct groups based on income and spending behavior")
    print("2. High-income customers show both high and low spending patterns")
    print("3. Low-income customers also show varied spending behaviors")
    print("4. The target segment (high income, high spending) represents valuable customers")
    print("5. Different marketing strategies can be developed for each cluster")
    
    # Save results
    output_file = "customer_segmentation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()