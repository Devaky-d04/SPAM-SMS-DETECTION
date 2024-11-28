# **Model Evaluation for SMS Spam Classification**

# Predictions using the trained Voting Classifier
y_pred = voting_clf.predict(X_test_tfidf)

# Accuracy and Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# **Confusion Matrix**
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# **ROC Curve**
fpr, tpr, _ = roc_curve(y_test, voting_clf.predict_proba(X_test_tfidf)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# **Precision-Recall Curve**
precision, recall, _ = precision_recall_curve(y_test, voting_clf.predict_proba(X_test_tfidf)[:, 1])
avg_precision = average_precision_score(y_test, voting_clf.predict_proba(X_test_tfidf)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (area = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# **SHAP Value Visualization for Feature Importance in XGBoost Model**
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_train_smote)
shap.summary_plot(shap_values, X_train_smote)
# **Final Prediction Function**
def predict_spam(message):
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = voting_clf.predict(message_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

new_message = "Congratulations! You've won a free ticket. Call now!"
print(f"Message: '{new_message}' is classified as:", predict_spam(new_message))

new_message2 = "Hey, can we meet at 6 pm for the party?"
print(f"Message: '{new_message2}' is classified as:", predict_spam(new_message2))

