import pickle

# Ucitavanje modela

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Ucitavanje TF-IDF vektorizatora

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("=== Product Category Predictor ===")

while True:

    product = input(
        "\nUnesi naziv proizvoda (ili exit): "
    )

    if product.lower() == "exit":
        break

    vector = vectorizer.transform(
        [product]
    )

    prediction = model.predict(
        vector
    )

    print(
        f"Predlozena kategorija: {prediction[0]}"
    )