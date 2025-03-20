import pickle

# Charger le modèle
with open("lightgbm_model ML OK.pkl", "rb") as file:
    model = pickle.load(file)

# Essayer d'extraire les features
if hasattr(model, "feature_name"):
    print("Features du modèle :", model.feature_name())
else:
    print("Impossible d'extraire les features.")