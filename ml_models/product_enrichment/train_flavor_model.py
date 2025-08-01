import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

INPUT_CSV = "ml_models/product_enrichment/labeled_flavor_products.csv"
MODEL_PATH = "ml_models/product_enrichment/flavor_model.pkl"
VECTORIZER_PATH = "ml_models/product_enrichment/flavor_vectorizer.pkl"

df = pd.read_csv(INPUT_CSV)
df.dropna(subset=['product_name', 'flavor'], inplace=True)

X = df['product_name']
y = df['flavor']

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vec, y)

with open(MODEL_PATH, 'wb') as f_model:
    pickle.dump(model, f_model)
with open(VECTORIZER_PATH, 'wb') as f_vec:
    pickle.dump(vectorizer, f_vec)

print("Flavor model trained and saved.")
