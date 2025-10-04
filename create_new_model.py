import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Wczytaj dane
df = pd.read_csv("welcome_survey_simple_v1.csv", sep=";")

# Zakładamy, że dane są już zakodowane (np. one-hot), jeśli nie — trzeba je przekształcić
df_encoded = pd.get_dummies(df)
df_encoded = df_encoded.fillna(0)

# Stwórz model
model = KMeans(n_clusters=3, random_state=42)
model.fit(df_encoded)

# Zapisz model do pliku
with open("welcome_survey_clustering_pipeline_v2.pkl", "wb") as f:
    pickle.dump(model, f)