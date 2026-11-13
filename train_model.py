import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Ucitavanje podataka

df = pd.read_csv("products.csv")

# Ciscenje podataka
df.dropna(
    subset=["Product Title", " Category Label"],
    inplace=True
)

# Ulazne i ciljne vrijednosti

X = df["Product Title"]
y = df[" Category Label"]

# Podjela podataka

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Vektorizacija teksta

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

# Treniranje modela

model = LinearSVC()

model.fit(
    X_train_vec,
    y_train
)

# Cuvanje modela
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model uspjesno sacuvan.")