# ----------------------------------------------------------
# 📌 Imports
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------------------------------------
# 📌 Load Data
# ----------------------------------------------------------
df = pd.read_csv('phq9_dataset.csv')
print("✅ Data loaded.")

# ----------------------------------------------------------
# 📌 Select relevant columns
# ----------------------------------------------------------
needed_cols = [
    'phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9',
    'age', 'sex', 'happiness.score', 'period.name'
]
df = df[needed_cols]
print("✅ Columns selected:", list(df.columns))

# ----------------------------------------------------------
# 📌 Drop rows with missing PHQ-9 items
# ----------------------------------------------------------
df = df.dropna(subset=['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9'])
print(f"✅ Rows remaining after PHQ drop: {df.shape[0]}")

# ----------------------------------------------------------
# 📌 Fill missing age
# ----------------------------------------------------------
df['age'] = df['age'].fillna(df['age'].median())

# ----------------------------------------------------------
# 📌 Clean 'sex' column
# ----------------------------------------------------------
df['sex'] = df['sex'].str.lower().replace({
    'm': 'male', 'f': 'female'
})
df['sex'] = df['sex'].where(df['sex'].isin(['male','female']), 'non-binary')
df['sex'] = df['sex'].fillna('non-binary')

print(f"✅ 'sex' column standardized. Unique values: {df['sex'].unique()}")

# ----------------------------------------------------------
# 📌 Create Depression Risk target
# ----------------------------------------------------------
df['phq_sum'] = df[['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9']].sum(axis=1)
df['Depression_Risk'] = (df['phq_sum'] >= 10).astype(int)

print(df[['phq_sum','Depression_Risk']].head())

# ----------------------------------------------------------
# 📌 Define features and target
# ----------------------------------------------------------
features = [
    'phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9',
    'age', 'sex', 'happiness.score', 'period.name'
]
X = df[features]
y = df['Depression_Risk']

# ----------------------------------------------------------
# 📌 Split train/test
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Split complete:", X_train.shape, X_test.shape)

# ----------------------------------------------------------
# 📌 Define preprocessing
# ----------------------------------------------------------
numeric_features = [
    'phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9',
    'age', 'happiness.score'
]
categorical_features = ['sex', 'period.name']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ----------------------------------------------------------
# 📌 Build pipeline
# ----------------------------------------------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("✅ Pipeline created.")

# ----------------------------------------------------------
# 📌 Train model
# ----------------------------------------------------------
pipeline.fit(X_train, y_train)
print("✅ Model trained!")

# ----------------------------------------------------------
# 📌 Evaluate model
# ----------------------------------------------------------
y_pred = pipeline.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 📌 Save trained model
# ----------------------------------------------------------
joblib.dump(pipeline, 'depression_model.pkl')
print("✅ Model saved as depression_model.pkl")

# ----------------------------------------------------------
# 📌 Save feature order for deployment
# ----------------------------------------------------------
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()

with open('feature_order.json', 'w') as f:
    json.dump(feature_names, f)

print("✅ Feature order saved as feature_order.json")
