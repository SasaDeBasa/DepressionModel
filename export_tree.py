import joblib
from sklearn.tree import export_graphviz
import os

# ----------------------------------------------------------
# 1️⃣ Load trained model
# ----------------------------------------------------------
pipeline = joblib.load('depression_model.pkl')
print("[✅] Model loaded!")

# ----------------------------------------------------------
# 2️⃣ Extract one tree from Random Forest
# ----------------------------------------------------------
rf = pipeline.named_steps['classifier']
single_tree = rf.estimators_[0]
print(f"[✅] Extracted 1 tree out of {len(rf.estimators_)}")

# ----------------------------------------------------------
# 3️⃣ Get feature names after preprocessing
# ----------------------------------------------------------
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
print("[✅] Feature names extracted:")
for f in feature_names:
    print("  -", f)

# ----------------------------------------------------------
# 4️⃣ Export to DOT file
# ----------------------------------------------------------
export_graphviz(
    single_tree,
    out_file='tree.dot',
    feature_names=feature_names,
    class_names=['Not Depressed', 'Depressed'],
    rounded=True,
    filled=True,
    special_characters=True
)
print("[✅] Exported to tree.dot")

# ----------------------------------------------------------
# 5️⃣ OPTIONAL: Try rendering to PDF/PNG if Graphviz is installed
# ----------------------------------------------------------
print("[ℹ️] Attempting to convert .dot to .pdf and .png using system Graphviz...")

if os.system('dot -V') == 0:
    # Convert to PDF
    os.system('dot -Tpdf tree.dot -o tree.pdf')
    print("[✅] Created tree.pdf")
    
    # Convert to PNG
    os.system('dot -Tpng tree.dot -o tree.png')
    print("[✅] Created tree.png")

else:
    print("[⚠️] Graphviz 'dot' command not found. Install Graphviz and ensure it's on PATH.")
    print("    Then run manually:")
    print("    dot -Tpdf tree.dot -o tree.pdf")
