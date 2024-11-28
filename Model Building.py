# **Splitting SMS Spam Dataset into Training and Testing Sets**
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

# **Vectorization**
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# **SMOTE for Balancing the Dataset**
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
print(pd.Series(y_train_smote).value_counts())

# **Ensemble SMS Spam Classification with Voting Classifier**
nb = MultinomialNB()
lr = LogisticRegression(class_weight='balanced', random_state=42)
svm = SVC(class_weight='balanced', probability=True, random_state=42)
xgb = XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('naive_bayes', nb),
    ('logistic_regression', lr),
    ('svm', svm),
    ('xgboost', xgb)
], voting='soft')

voting_clf.fit(X_train_smote, y_train_smote)

# **Hyperparameter Tuning for XGBoost**
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)
best_xgb = grid_search.best_estimator_

voting_clf = VotingClassifier(estimators=[
    ('naive_bayes', nb),
    ('logistic_regression', lr),
    ('svm', svm),
    ('xgboost', best_xgb)
], voting='soft')
voting_clf.fit(X_train_smote, y_train_smote)
