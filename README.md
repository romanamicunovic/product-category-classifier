Product Category Classification


Opis projekta

Ovaj projekat koristi mašinsko učenje za automatsku klasifikaciju proizvoda na osnovu naziva proizvoda.

Model je treniran koristeći skup podataka products.csv i predviđa kojoj kategoriji pripada novi proizvod.

Korištene tehnologije

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- LinearSVC

Struktura projekta

- product_classification.ipynb - analiza i razvoj modela
- train_model.py - treniranje i čuvanje modela
- predict_category.py - interaktivno testiranje modela
- products.csv - skup podataka
- model.pkl - sačuvan model
- vectorizer.pkl - sačuvan TF-IDF vektorizer

Pokretanje treninga
python train_model.py

Pokretanje predikcije
python predict_category.py

Primjer korištenja
Unos:
iphone 7 32gb gold
Rezultat:
Mobile Phones

Autor
Projekat izrađen u okviru kursa Introduction to Machine Learning Using Python


