# Stress Detection using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load sample dataset (replace 'sample_data.csv' with your dataset)
data = pd.read_csv('sample_data.csv')  # columns: EDA, HR, label

# 2. Features and Labels
X = data[['EDA', 'HR']]  # example features
y = data['label']          # 0 = No Stress, 1 = Stress

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Predict
y_pred = clf.predict(X_test)

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Example prediction for a new user sample
new_sample = [[0.5, 72]]  # Example: EDA=0.5, HR=72
new_sample_scaled = scaler.transform(new_sample)
prediction = clf.predict(new_sample_scaled)
print(f"Prediction for new sample: {'Stress' if prediction[0]==1 else 'No Stress'}")
