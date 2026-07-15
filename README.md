# OptiCrop - Smart Agricultural Production Optimization Engine

OptiCrop is an AI-powered smart agricultural recommendation system designed to support modern, data-driven farming practices. The system analyzes critical soil and environmental parameters to recommend the most viable and productive crops for cultivation. 

This project is built using Python, Flask, Pandas, and Scikit-learn, ready to be pushed to GitHub to complete the **Skill Wallet** task.

---

## 🌟 Project Architecture & Workflow

OptiCrop follows a structured, modular machine learning pipeline:
1. **Data Collection**: Retrieves historical crop performance data (`Crop_recommendation.csv`) measuring nitrogen, phosphorous, potassium, temperature, humidity, pH, and rainfall.
2. **Exploratory Data Analysis & Cleaning**: Scans for missing observations and handles statistical outliers using the **Interquartile Range (IQR)** method.
3. **Unsupervised Analysis (K-Means)**: Groups crops into clusters based on soil and climatic attributes to discover functional profiles.
4. **Supervised Classification (Logistic Regression)**: Fits a 22-class classification model, yielding **98% prediction accuracy**.
5. **Serialization**: Exports the trained classifier to `model.pkl` for low-latency web inference.
6. **Web Interface**: A premium dark-forest glassmorphic web dashboard where users input soil data to query the recommendation engine.

---

## 🛠️ Technology Stack
*   **Backend**: Python 3.x, Flask (WSGI Server)
*   **Machine Learning**: Scikit-learn, Pandas, NumPy, SciPy
*   **Data Visualization**: Matplotlib, Seaborn
*   **Frontend**: HTML5, Vanilla CSS3 (Modern Glassmorphism, Responsive Grid, Google Fonts)

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your computer.

### 2. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to download the dataset, preprocess it, train the models, and serialize the best model:
```bash
python train.py
```
This generates a file named `model.pkl` in your project folder.

### 4. Launch the Web Application
Start the local Flask development server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the application.

### 5. Run Verification Tests
To run the automated suite testing page routes and prediction validity:
```bash
python test_app.py
```

---

## 🤝 Team Members
*   👑 **Nehan Chowdary Muvva** (Team Lead)
*   👨‍💻 **Jaswanth Malapareddy**
*   👨‍💻 **Madasu Rushi Venkata Ganesh**
*   👩‍💻 **Rajani Meesala**
*   👩‍💻 **Mounika Sonti**

---

## 📤 Submitting to GitHub
To push this project to your GitHub repository:

1. Create a new repository on [GitHub](https://github.com/new) called `OptiCrop`.
2. Open your terminal in this directory and execute:
   ```bash
   # Add all files to staging
   git add .
   
   # Commit changes
   git commit -m "Initial commit: Completed OptiCrop ML model and Flask web app"
   
   # Add your remote repository (Replace with your actual repo link)
   git remote add origin https://github.com/YOUR_USERNAME/OptiCrop.git
   
   # Push to the main branch
   git branch -M main
   git push -u origin main
   ```
3. Submit the GitHub URL to complete your Skill Wallet project!
