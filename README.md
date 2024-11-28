# SPAM-SMS-DETECTION
This project uses a classification model to identify spam messages in the SMS Spam Collection dataset. The purpose is to label messages as "spam" or "ham" (non-spam). The project detects spam using a variety of machine learning algorithms, ensemble methodologies, and a neural network (LSTM) model.

**Key Features:**

Ensemble Learning: Combines multiple classifiers to improve classification performance.
SMOTE: Balances the dataset by generating synthetic samples to handle class imbalance.
Neural Network (LSTM): Uses a deep learning model to capture text sequence patterns for spam detection.
Model Evaluation: Includes comprehensive evaluation metrics like accuracy, confusion matrix, ROC curve, precision-recall curve, and SHAP values for feature importance.

**Libraries Used:**

Data Processing and Analysis: pandas, numpy, scikit-learn
Machine Learning Models: scikit-learn, xgboost, imblearn
Deep Learning: TensorFlow, Keras
Visualization: matplotlib, seaborn
Model Interpretability: shap

**Requirements
To run this project, you need the following libraries:**

pandas
numpy
scikit-learn
xgboost
imblearn
matplotlib
seaborn
tensorflow
shap

**Running the Project:**
1. Load the Dataset
The SMSSpamCollection dataset is read, and the labels are encoded into binary format (ham=0, spam=1).

2. Preprocess the Data
The text data is vectorized using TF-IDF.
SMOTE is applied to balance the dataset for training.

3. Train the Models
The models trained include Naive Bayes, Logistic Regression, SVM, XGBoost, and an LSTM neural network.

4. Evaluation
The models are evaluated on test data with metrics like accuracy, precision, recall, F1 score, ROC curve, and confusion matrix.

5. Use the Prediction Function
The predict_spam(message) function classifies any input message as spam or ham.
