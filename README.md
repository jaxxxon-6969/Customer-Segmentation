# Customer Segmentation Using K-Means Clustering

## 📌 Project Overview

This project performs **customer segmentation using Machine Learning**. The goal is to group customers into different segments based on their **Annual Income** and **Spending Score**.

The project uses the **K-Means Clustering** algorithm to identify groups of customers with similar spending behavior.

---

## 🎯 Objectives

* Understand and explore customer data
* Check the dataset for missing values and duplicate records
* Perform basic Exploratory Data Analysis (EDA)
* Select relevant features for clustering
* Scale the selected features
* Determine the appropriate number of clusters using the **Elbow Method**
* Apply **K-Means Clustering**
* Visualize the resulting customer segments
* Assign a new customer to an existing cluster

---

## 📊 Dataset

The project uses the **Mall Customers dataset** stored as:

```text
Mall_Customers.csv
```

The dataset contains **200 customers** and 5 columns:

| Column                   | Description                           |
| ------------------------ | ------------------------------------- |
| `CustomerID`             | Unique customer identifier            |
| `Genre`                  | Customer gender                       |
| `Age`                    | Customer age                          |
| `Annual Income (k$)`     | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Customer spending score               |

The clustering model uses the following two features:

* **Annual Income (k$)**
* **Spending Score (1-100)**

---

## 🛠️ Technologies & Libraries

The project is implemented in Python using:

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook / Google Colab

### Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

---

## 🔄 Project Workflow

The project follows these steps:

```text
Load Dataset
     ↓
Understand Dataset
     ↓
Check Missing Values
     ↓
Check & Remove Duplicates
     ↓
Exploratory Data Analysis
     ↓
Select Features
     ↓
Feature Scaling
     ↓
Elbow Method
     ↓
K-Means Clustering
     ↓
Add Cluster Labels
     ↓
Visualize Customer Segments
     ↓
Analyze Clusters
     ↓
Predict Cluster for New Customer
```

---

## 🔍 Exploratory Data Analysis

The notebook explores the distributions of:

### Age

A histogram is created to understand the age distribution of customers.

### Annual Income

The annual income distribution is visualized using a histogram.

### Spending Score

The spending score distribution is also visualized to understand customer spending behavior.

---

## ⚙️ Data Preprocessing

The dataset is checked for missing values:

```python
df.isnull().sum()
```

The notebook also checks for duplicate records:

```python
df.duplicated().sum()
```

Duplicate records are removed using:

```python
df = df.drop_duplicates()
```

The dataset contains **200 records**, with no missing values and no duplicate records in the provided dataset.

---

## 📐 Feature Selection

For customer segmentation, the following features are selected:

```python
x = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]
```

These features are used because they provide a useful representation of customers' income and spending behavior.

---

## 📏 Feature Scaling

Before applying K-Means, the selected features are standardized using `StandardScaler`:

```python
scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
```

Scaling is important because the K-Means algorithm uses distance calculations.

---

## 📈 Elbow Method

The **Elbow Method** is used to examine different values of `K`.

```python
WCSS = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    model.fit(x_scaled)
    WCSS.append(model.inertia_)
```

The notebook plots WCSS against the number of clusters to help determine an appropriate value of `K`.

---

## 🤖 K-Means Clustering

The notebook applies K-Means clustering with **5 clusters**:

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(x_scaled)
```

The resulting cluster labels are added to the original dataset:

```python
df['Cluster'] = clusters
```

---

## 📊 Cluster Visualization

The customer segments are visualized using a scatter plot:

```python
plt.scatter(
    x_scaled[:, 0],
    x_scaled[:, 1],
    c=clusters
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Using K-Means")
plt.show()
```

This visualization helps identify groups of customers with similar income and spending-score characteristics.

---

## 👤 New Customer Prediction

The project also demonstrates how a new customer can be assigned to an existing cluster.

Example customer:

```python
new_customer = np.array([[80, 75]])
```

The customer data is scaled and passed to the trained K-Means model:

```python
new_customer_scaled = scaler.transform(new_customer)

cluster = kmeans.predict(new_customer_scaled)

print("Customer belongs to cluster", cluster[0])
```

For the example used in the notebook, the model assigns the customer to **Cluster 1**.

---

## 📁 Project Structure

```text
Customer-Segmentation/
│
├── Customer_segmenation.ipynb
├── Mall_Customers.csv
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Navigate to the project folder

```bash
cd Customer-Segmentation
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Open the notebook

```bash
jupyter notebook Customer_segmenation.ipynb
```

You can also open the notebook directly using **Google Colab**.

---

## 📌 Key Machine Learning Concepts

This project demonstrates:

* Unsupervised Learning
* Customer Segmentation
* K-Means Clustering
* Feature Scaling
* Exploratory Data Analysis
* Elbow Method
* WCSS (Within-Cluster Sum of Squares)
* Cluster Visualization
* New Data Prediction

---

## 🔮 Possible Future Improvements

The project can be extended by:

* Adding more customer features such as age and gender
* Comparing different clustering algorithms
* Using Silhouette Score to evaluate clustering
* Creating detailed cluster profiles
* Building an interactive dashboard using Power BI or Streamlit
* Creating a customer-segmentation web application
* Saving and loading the trained clustering model

---

## 👨‍💻 Author

**Jackson Abraham**

This project was created as a Machine Learning practice project to understand **customer segmentation and unsupervised learning using Python and Scikit-learn**.

---

## ⭐ If You Find This Project Useful

Feel free to ⭐ star the repository and explore the notebook to understand how K-Means clustering can be used for customer segmentation.
