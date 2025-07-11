# ----------------------------------------------------------
# 📌 Imports
# ----------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ----------------------------------------------------------
# 📌 Step 1: Load Data
# ----------------------------------------------------------
df = pd.read_csv('synthetic_phq9_dataset.csv')
print("Data loaded!")

# ----------------------------------------------------------
# 📌 Step 2: Select relevant columns
# ----------------------------------------------------------
needed_cols = ['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9',
               'age', 'sex', 'happiness.score', 'period.name']

df = df[needed_cols]
print("Selected columns:", df.columns.tolist())

# ----------------------------------------------------------
# 📌 Step 3: Clean missing values
# ----------------------------------------------------------
# In synthetic data you may have no NaNs, but this is safe
df = df.dropna(subset=['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9'])
df['age'] = df['age'].fillna(df['age'].median())
df['sex'] = df['sex'].fillna('non-binary')
df['happiness.score'] = df['happiness.score'].fillna(df['happiness.score'].median())
df['period.name'] = df['period.name'].fillna('unknown')
print("Cleaned missing values.")

# ----------------------------------------------------------
# 📌 Step 4: Standardize sex column
# ----------------------------------------------------------
df['sex'] = df['sex'].str.lower().replace({
    'm': 'male',
    'f': 'female',
})
df['sex'] = df['sex'].where(df['sex'].isin(['male','female']), 'non-binary')
print("Standardized sex values.")

# ----------------------------------------------------------
# 📌 Step 5: Create target variable
# ----------------------------------------------------------
df['phq_sum'] = df[[f'phq{i}' for i in range(1,10)]].sum(axis=1)
df['Depression_Risk'] = (df['phq_sum'] >= 10).astype(int)
print(df[['phq_sum', 'Depression_Risk']].head())

# ----------------------------------------------------------
# 📌 Step 6: Define Features and Target
# ----------------------------------------------------------
features = ['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9',
            'age', 'sex', 'happiness.score', 'period.name']
X = df[features]
y = df['Depression_Risk']

# ----------------------------------------------------------
# 📌 Step 7: Train-test split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ----------------------------------------------------------
# 📌 Step 8: Preprocessing
# ----------------------------------------------------------
numeric_features = ['phq1','phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9',
                     'age', 'happiness.score']
categorical_features = ['sex', 'period.name']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ----------------------------------------------------------
# 📌 Step 9: Pipeline
# ----------------------------------------------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
print("Pipeline built.")

# ----------------------------------------------------------
# 📌 Step 10: Train model
# ----------------------------------------------------------
pipeline.fit(X_train, y_train)
print("Model trained!")

# ----------------------------------------------------------
# 📌 Step 11: Evaluate
# ----------------------------------------------------------
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 📌 Step 12: Save Model
# ----------------------------------------------------------
joblib.dump(pipeline, 'depression_model.pkl')
print("Model saved as depression_model.pkl")
