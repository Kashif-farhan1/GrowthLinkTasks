# 🔍 Machine Learning and Deep Learning Projects Collection

This repository contains five well-documented machine learning and deep learning projects implemented in Python using Jupyter/Google Colab. Each project covers a different use case with relevant datasets, preprocessing steps, modeling, evaluation metrics, and visualizations.

---

## 📁 Project Files

| File Name                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `customer_churn_prediction.ipynb` | Predict whether a customer is likely to leave a service or product.         |
| `fraud_detection_task2.ipynb`     | Detect fraudulent transactions using machine learning algorithms.           |
| `handwritten_text_generation.ipynb` | Generate realistic handwritten-style text using deep learning techniques. |
| `movie_genre_classification.ipynb` | Classify movie genres based on title and plot using NLP and ML models.     |
| `spam_sms_detection_colab.ipynb`  | Detect spam messages in SMS texts using NLP and classification models.     |

---

## 📌 Task Details

### 1️⃣ Customer Churn Prediction
- **Objective**: Predict customer churn based on usage patterns.
- **Techniques Used**: Logistic Regression, Random Forest, XGBoost, Data Preprocessing.
- **Evaluation Metrics**: Accuracy, Confusion Matrix, ROC-AUC.
- ** Dataset **: "https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction".

### 2️⃣ Credit Card Fraud Detection
- **Objective**: Build a model to identify fraudulent transactions.
- **Techniques Used**: Anomaly Detection, Class Balancing, Decision Trees.
- **Evaluation Metrics**: Precision, Recall, F1-Score, AUC.
- ** Dataset **: "https://www.kaggle.com/datasets/kartik2112/fraud-detection".

### 3️⃣ Handwritten Text Generation
- **Objective**: Generate handwritten-style text using deep learning.
- **Dataset**: `train_v2.zip`, `test_v2.zip`, `validate_v2.zip` (JPG images).
- **Techniques Used**: Vision Transformers (TrOCR), Image Preprocessing, OCR Decoding.
- **Evaluation Metrics**: Visual inspection of generated handwriting vs. ground truth.
- ** Dataset **: "https://www.kaggle.com/code/ahmedtoba/handwritten-text-generation/input".

### 4️⃣ Movie Genre Classification
- **Objective**: Classify movie genres using title and plot descriptions.
- **Dataset Format**: `ID ::: Title ::: Genre ::: Plot`.
- **Techniques Used**: NLP (TF-IDF), Multinomial Naive Bayes, Logistic Regression.
- **Evaluation Metrics**: Accuracy, F1-Score, Classification Report.
- ** Dataset **: "https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb".

### 5️⃣ Spam SMS Detection
- **Objective**: Classify SMS messages as Spam or Ham.
- **Techniques Used**: NLP preprocessing, TF-IDF Vectorization, Naive Bayes.
- **Evaluation Metrics**: Accuracy, ROC Curve, AUC Score.
-  ** Dataset **: "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset".

---

## 📦 Requirements
- Python 3.x
- Jupyter Notebook / Google Colab
- scikit-learn, pandas, numpy, matplotlib, seaborn
- transformers (for TrOCR - handwritten text generation)
- cv2 / PIL for image handling

---

## 📂 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
2. upload files in google colab(if using).
3. Exeute each cell one by one.
4. Make sure that you have necessary Datasets and above requiremnets.
5. Make sure that you have added correct path for each task.


-----


## Data Sets:
1. For Movie genre classification use "https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb" the link and download dataset.
2. For Credit Card Fraud Detection use "(https://www.kaggle.com/datasets/kartik2112/fraud-detection)" the link and download dataset.
3. For Customer Churn Prediction use "https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction" the link and download dataset.
4. For Spam SMS Detection use "https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset" the link and download dataset.
5. For Handwritten Text Generation use "https://www.kaggle.com/code/ahmedtoba/handwritten-text-generation" the link and download dataset.
