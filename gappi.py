import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
# Load dataset
data_path = '/content/e850e4e5-8147-4373-8a1c-8200dcc0ebcc-loan-test (1).csv'
data = pd.read_csv(data_path)
print(data.head())
print(data.columns)
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Check if 'EmploymentStatus' exists in the DataFrame
if 'EmploymentStatus' in data.columns:
    label_encoder = LabelEncoder()
    data['EmploymentStatus'] = label_encoder.fit_transform(data['EmploymentStatus'])
else:
    print("Column 'EmploymentStatus' not found in the DataFrame.")
    # Investigate why the column is missing
    print(data.columns) # Print available columns to help identify the issue
    features = ['Credit_History', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']  # Update these names based on the actual column names you want to use
X = data[features]
y = data['Loan_ID']  # Update to the correct target column

# Encode target variable if necessary
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Now X contains the selected features and y_encoded contains the encoded target variable
print(X.head())
print(y_encoded[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)