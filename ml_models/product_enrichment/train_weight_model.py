import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Пути
INPUT_CSV = "ml_models/product_enrichment/labeled_weight_products.csv"
MODEL_PATH = "ml_models/product_enrichment/weight_model.pkl"
VECTORIZER_PATH = "ml_models/product_enrichment/weight_vectorizer.pkl"

# Загрузка размеченных данных
df = pd.read_csv(INPUT_CSV, sep=';')
df.dropna(subset=['product_name', 'weight'], inplace=True)

X = df['product_name']
y = df['weight'].astype(str)  # Преобразуем в строки, если модель классификации

# Обучение
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vec, y)

# Сохраняем модель и векторизатор
with open(MODEL_PATH, 'wb') as f_model:
    pickle.dump(model, f_model)

with open(VECTORIZER_PATH, 'wb') as f_vec:
    pickle.dump(vectorizer, f_vec)

print(f"✅ Модель веса обучена и сохранена:\n→ {MODEL_PATH}\n→ {VECTORIZER_PATH}")
