import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained pipeline
pipeline = joblib.load('depression_model.pkl')

# Extract preprocessing step
preprocessor = pipeline.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out()

print("Feature names:", feature_names)

# Extract model
model = pipeline.named_steps['classifier']
importances = model.feature_importances_

# Make DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Features:")
print(importance_df.head(10))

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(10),
    palette='viridis'
)
plt.title('Top 10 Feature Importances for Depression Risk Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300)
plt.show()
