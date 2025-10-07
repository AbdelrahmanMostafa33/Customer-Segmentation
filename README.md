# 🛍️ Customer Segmentation (Mall Customers)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Clustering-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-teal.svg)

Cluster mall customers based on their **Annual Income** and **Spending Score** to uncover actionable marketing insights.

---

## 📌 Overview

This project performs **customer segmentation** using unsupervised learning techniques.
It explores shopping behaviors to identify distinct customer groups and visualize their spending patterns.

**Main goals:**
- Clean and preprocess customer data
- Explore relationships between income and spending
- Apply and evaluate **K-Means** clustering
- Determine optimal number of clusters using **Elbow & Silhouette** methods
- Visualize the clusters and their centroids
- Experiment with **DBSCAN** as an alternative clustering algorithm

---

## 📂 Dataset

Dataset: [Mall Customers -- Kaggle](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

**Features include:**
- `CustomerID` --- unique identifier
- `Gender` --- male or female
- `Age` --- customer age
- `Annual Income (k$)` --- income in thousands
- `Spending Score (1-100)` --- a spending rating assigned by the mall

**Target:** No target (unsupervised learning).

---

## 🔑 Project Workflow

1. **Data Loading & Cleaning**
   - Removed duplicates and checked for missing values
   - Standardized column names
   - Kept relevant numerical features for clustering

2. **Exploratory Data Analysis (EDA)**
   - Histograms and KDEs for income and spending distribution
   - Boxplots to detect outliers
   - Scatter plots to visualize spending patterns

3. **Feature Scaling**
   - Standardized numerical features using `StandardScaler`

4. **Modeling with K-Means**
   - Used **Elbow Method** and **Silhouette Score** to determine optimal k
   - Chose **k = 5** clusters as optimal
   - Added cluster labels to dataset

5. **Cluster Analysis**
   - Calculated average income, age, and spending per cluster
   - Named customer segments for easier interpretation

   **Cluster Names:**
   | Cluster | Segment Name       | Description |
   |----------|-------------------|--------------|
   | 0 | Average Spenders | Middle-income, moderate spenders |
   | 1 | Luxury Shoppers | High-income, high-spending |
   | 2 | Careful Rich | High-income, low-spending |
   | 3 | Low Budget | Low-income, low-spending |
   | 4 | High Spenders | Low-income, high-spending |

6. **Bonus: DBSCAN**
   - Applied DBSCAN with `eps=0.5` and `min_samples=5`
   - Detected outliers (noise) and compact groups

7. **Visualization**
   - 2D scatter plots showing customer clusters
   - Centroid markers for K-Means groups
   - Legend replaced with readable cluster names

8. **Saving Results**
   - Exported labeled dataset to:
     ```
     data/processed/Mall_Customers_Segmented.csv
     ```

---

## 📊 Results & Insights

- **Optimal clusters:** 5
- **Distinct segments:** from low-budget to luxury spenders
- **K-Means** offered interpretable, balanced segments
- **DBSCAN** revealed natural groupings but was more sensitive to scaling and parameters
- Clear separation between **high-income cautious spenders** and **enthusiastic low-income spenders**

**Use Cases:**
- Targeted marketing campaigns
- Personalized promotions
- Customer loyalty programs

---

## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/AbdooMatrix/Customer-Segmentation.git
cd Customer-Segmentation
```

### 2\. Create a Virtual Environment

```
python -m venv venv
```

Activate it:

-   **Windows:**

    ```
    venv\Scripts\activate
    ```

-   **Mac/Linux:**

    ```
    source venv/bin/activate
    ```

### 3\. Install Dependencies

```
pip install -r requirements.txt
```

### 4\. Run the Project

```
python src/main.py
```

This will:

-   Load and clean the dataset

-   Perform EDA

-   Apply K-Means and DBSCAN

-   Visualize clusters and centroids

-   Save the final labeled dataset

* * * * *

🧠 Future Improvements
----------------------

-   Add **PCA** for dimensionality reduction

-   Compare **Agglomerative Clustering** performance

-   Include more customer behavior features (e.g., frequency, recency)

-   Build an interactive dashboard (e.g., Streamlit or Dash)

* * * * *

📁 Folder Structure
-------------------

```
Customer-Segmentation/
│
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv
│   └── processed/
│       └── Mall_Customers_Segmented.csv
│
├── src/
│   └── main.py
│
├── notebooks/
│   └── Customer_Segmentation.ipynb
│
├── requirements.txt
└── README.md
```

* * * * *

🏁 Conclusion
-------------

The project successfully segmented mall customers into **five meaningful groups** based on their income and spending habits.\
K-Means clustering provided clear, actionable insights that can guide **targeted marketing strategies** and **customer retention efforts**.\
This demonstrates the value of unsupervised learning for **data-driven decision-making** in retail.