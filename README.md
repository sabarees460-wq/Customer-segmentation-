# 🎯 Customer Segmentation & Analytics Dashboard

An interactive, beginner-friendly **Customer Segmentation and Analytics Dashboard** built in Python using **Streamlit**, **Pandas**, **Scikit-learn**, and **Plotly**. 

This project groups mall customers into distinct personas based on their **Annual Income** and **Spending Score** using **K-Means Clustering**, and automatically finds the optimal number of clusters using the **Elbow Method**.

---

## 🌟 Key Features

1. **Benchmark & Custom CSV Support**: Runs instantly with the included `Mall_Customers.csv` dataset, or allows uploading any custom customer CSV file.
2. **Intelligent Column Detection**: Auto-detects columns for Income, Spending Score, Age, Gender, and ID even if column headers differ across datasets.
3. **Data Cleaning & Scaling**: Automatically handles missing values, removes duplicate records, and scales numeric features using `StandardScaler` or `MinMaxScaler`.
4. **Automated Elbow Method**: Calculates WCSS (Within-Cluster Sum of Squares) and uses a vector distance knee-detection algorithm to determine the optimal number of clusters ($K$).
5. **Silhouette Analysis**: Evaluates clustering separation and density across different cluster counts.
6. **Customer Personas**: Dynamically assigns meaningful business personas:
   - 💎 **VIP / Target Spenders**: High Income, High Spending
   - 🏦 **Careful / Conservative Savers**: High Income, Low Spending
   - 🛍️ **Impulsive / High Spenders**: Low Income, High Spending
   - 🛡️ **Budget / Sensible Shoppers**: Low Income, Low Spending
   - ⚖️ **Standard Base**: Moderate Income, Moderate Spending
7. **Interactive 2D & 3D Plotly Visualizations**: Hover cards, cluster centroid markers, and 3D feature views.
8. **Live Customer Segment Predictor**: Form for typing custom customer values to instantly predict their segment.
9. **Data Export**: Download segmented dataset with added `Cluster` and `Persona` columns as CSV.
10. **Modern College-Project UI**: Metric KPI cards, tabbed navigation, custom CSS styling, and project defense explainers.

---

## 📂 Project Structure

```text
Customer_Segmentation/
├── Mall_Customers.csv         # Standard benchmark dataset (200 rows)
├── app.py                     # Main Streamlit dashboard application
├── requirements.txt           # Python dependencies
├── README.md                  # Complete project documentation & guide
└── utils/
    ├── __init__.py
    ├── data_processor.py      # Data loading, column mapping, cleaning & scaling
    ├── clustering.py          # K-Means, Elbow vector locator & persona generator
    └── visualizations.py      # Interactive Plotly 2D/3D charts & heatmaps
```

---

## 🛠️ Step-by-Step Setup & Run Instructions

### Prerequisites
Make sure you have **Python 3.9** or higher installed on your computer.

### Step 1: Open Terminal / Command Prompt
Navigate to the project root directory:
```bash
cd d:/Customer_Segmentation
```

### Step 2: Install Required Packages
Install all required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Dashboard
Launch the dashboard app:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧮 Mathematical & Technical Explanation

### 1. K-Means Clustering Algorithm
K-Means partitions $N$ observations into $K$ clusters in which each observation belongs to the cluster with the nearest mean (centroid).
- **Objective Function**: Minimizes the Within-Cluster Sum of Squares (WCSS):
  $$WCSS = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

### 2. Automatic Elbow Detection Algorithm
To find the knee point on the WCSS curve:
- We construct a line segment connecting $(K_{min}, WCSS_{min})$ to $(K_{max}, WCSS_{max})$.
- For each point $(K_i, WCSS_i)$, we compute its perpendicular distance to this line segment:
  $$d_i = \frac{|(y_n - y_1) x_i - (x_n - x_1) y_i + x_n y_1 - y_n x_1|}{\sqrt{(y_n - y_1)^2 + (x_n - x_1)^2}}$$
- The cluster count $K$ that maximizes $d_i$ is selected as the optimal **Elbow Point**.

---

## 🧪 Testing & Verification

Run headless Python checks to ensure all modules work without errors:
```bash
python -c "from utils.data_processor import load_data; print(load_data('Mall_Customers.csv')[0].shape)"
```

---

## 🎓 Suitable for College Projects & Submissions
This project includes complete modular code, defensive error handling, inline docstrings, intuitive metric dashboards, and mathematical explanations perfect for project reports and viva presentations.
