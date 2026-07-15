import os
import requests
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# 1. Download the dataset
dataset_path = 'Crop_recommendation.csv'
dataset_url = 'https://raw.githubusercontent.com/gabbygab1233/Crop-Recommender/main/Crop_recommendation.csv'
backup_url = 'https://raw.githubusercontent.com/amal-shaji/Crop-Recommendation-Dataset/main/Crop_recommendation.csv'

if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    try:
        response = requests.get(dataset_url)
        response.raise_for_status()
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        print("Dataset downloaded successfully from primary source.")
    except Exception as e:
        print(f"Primary source failed: {e}. Trying backup source...")
        try:
            response = requests.get(backup_url)
            response.raise_for_status()
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            print("Dataset downloaded successfully from backup source.")
        except Exception as e2:
            print(f"Backup source failed: {e2}.")
            raise RuntimeError("Could not download the Crop Recommendation dataset.")
else:
    print("Dataset already exists locally.")

# 2. Load the dataset
df = pd.read_csv(dataset_path)
print("Initial dataset shape:", df.shape)
print("Columns in dataset:", df.columns.tolist())

# Rename columns to match the spelling in the PDF (nitrogen, phosphorous, potassium)
rename_map = {'N': 'nitrogen', 'P': 'phosphorous', 'K': 'potassium'}
df = df.rename(columns=rename_map)
print("Columns after renaming:", df.columns.tolist())

# 3. Check for Null Values (as per PDF Page 16)
print("\nChecking for Null Values:")
print(df.isnull().sum())

# 4. Outlier Handling (as per PDF Page 15)
# Outliers are identified using IQR for the phosphorous column
print("\nHandling outliers for 'phosphorous' feature...")
Q1 = df['phosphorous'].quantile(0.25)
Q3 = df['phosphorous'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filter_condition = (df['phosphorous'] >= lower_bound) & (df['phosphorous'] <= upper_bound)
df = df.loc[filter_condition]
print("Dataset shape after outlier filtering:", df.shape)

# 5. K-Means Clustering Analysis (as per PDF Page 11)
print("\nRunning K-Means Clustering on features to inspect groups...")
features_only = df.drop(['label'], axis=1)
km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(features_only)

z = pd.concat([pd.DataFrame(y_means, columns=['cluster']), df['label'].reset_index(drop=True)], axis=1)
for i in range(4):
    cluster_crops = z[z['cluster'] == i]['label'].unique()
    print(f"Crops in Cluster {i+1}:", cluster_crops)

# 6. Train-Test Split (as per PDF Page 13)
y = df['label']
X = df.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# 7. Model Training: Logistic Regression (as per PDF Page 10)
# We set max_iter to 2000 to ensure convergence
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=2000, random_state=0)
model.fit(X_train, y_train)

# 8. Model Evaluation (as per PDF Page 9)
y_pred = model.predict(X_test)
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))

# Test predict call matching PDF Page 8
test_input = np.array([[105, 35, 40, 25, 64, 7, 160]])
test_pred = model.predict(test_input)
print(f"Test prediction for N=105, P=35, K=40, Temp=25, Humid=64, pH=7, Rain=160 -> {test_pred[0]}")

# 9. Serialize Model (as per PDF Page 9 & Page 4)
model_file = 'model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved successfully as '{model_file}'.")
