import joblib
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

# Load the trained pipeline
pipeline = joblib.load('depression_model.pkl')

# 1️⃣ Get the preprocessor and the trained RandomForestClassifier
preprocessor = pipeline.named_steps['preprocessor']
rf_model = pipeline.named_steps['classifier']

# 2️⃣ Get the feature names after preprocessing
# This will handle both numeric and one-hot-encoded categorical features
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
all_features = list(num_features) + list(cat_features)

print("All feature names after preprocessing:")
print(all_features)

# 3️⃣ Choose one tree to visualize (e.g. the first one)
estimator = rf_model.estimators_[0]

# 4️⃣ Plot the tree using matplotlib
plt.figure(figsize=(30, 15))
tree.plot_tree(
    estimator,
    feature_names=all_features,
    class_names=['No Risk', 'Risk'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Visualization of One Decision Tree in the RandomForest")
plt.show()

# 5️⃣ Optionally save the tree as an image
plt.savefig("decision_tree_visualization.png", dpi=300)
