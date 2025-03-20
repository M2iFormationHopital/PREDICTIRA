from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np


from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Active CORS pour accepter les requêtes de l'extérieur


# Charger le fichier CSV
csv_path = "df_Test_Indust.csv"
df = pd.read_csv(csv_path)

# Charger le modèle Machine Learning
model_path = "lightgbm_model ML OK.pkl"
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Modèle chargé avec succès")
except Exception as e:
    print(" Erreur lors du chargement du modèle :", e)

# Liste des features attendues par le modèle
# features = [
#     'SYM1', 'SYM22', 'SYM23', 'SYM29', 'SYM3', 'SYM31', 'SYM34', 'SYM35', 'SYM5',
#     'SYM6', 'SYM68', 'SYM8', 'average_Co', 'average_IQA_global', 'average_No_2',
#     'average_O_3', 'average_Pm_2_5', 'average_So_2', 'avg_pressure',
#     'avg_temperature_max', 'dept', 'indice', 'week_cos', 'year'
# ]


features = [
    {"name": "SYM1", "label": "Symptome Douleur thoracique :", "tooltip": "Tendance de recherche des mots Douleur thoracique sur Google"},
    {"name": "SYM22", "label": "Symptome Faiblesse :", "tooltip": "Tendance de recherche du mot Faiblesse sur Google"},
    {"name": "SYM23", "label": "Symptome Toux :", "tooltip": "Tendance de recherche du mot Toux sur Google"},
    {"name": "SYM29", "label": "Symptome Sifflements :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche du mot Sifflements sur Google"},
    {"name": "SYM3", "label": "Symptome Essoufflement :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche du mot Essoufflement sur Google"},
    {"name": "SYM31", "label": "Symptome Production de mucus :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche des mots Production de mucus sur Google"},
    {"name": "SYM34", "label": "Symptome Maux de tête :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche des mots Maux de tête sur Google"},
    {"name": "SYM35", "label": "Symptome Perte d'appétit :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche des mots Perte d'appétit sur Google"},
    {"name": "SYM5", "label": "Symptome Nausées :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche du mot Nausées sur Google"},
    {"name": "SYM6", "label": "Symptome Vomissements :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche du mot Vomissements sur Google"},
    {"name": "SYM68", "label": "Symptome Frissons :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche du mot Frissons sur Google"},
    {"name": "SYM8", "label": "Symptome Palpitations :", "tooltip": "Renseigner la valeur du symptome : Tendance de recherche du mot Palpitations sur Google"},
    {"name": "average_Co", "label": "Moyenne Monoxyde de carbone :", "tooltip": "Valeur moyenne du monoxyde de carbone sur la semaine de prédiction"},
    {"name": "average_IQA_global", "label": "Moyenne Indices polluants :", "tooltip": "Moyenne des indices de dangerosité des polluants sur la semaine de prédiction"},
    {"name": "average_No_2", "label": "Moyenne Dioxyde d'azote :", "tooltip": "Valeur moyenne du dioxyde d'azote sur la semaine de prédiction"},
    {"name": "average_O_3", "label": "Moyenne Ozone :", "tooltip": "Valeur moyenne de l'ozone sur la semaine de prédiction"},
    {"name": "average_Pm_2_5", "label": "Moyenne particules très fines (PM 2.5) :", "tooltip": "Valeur Moyenne des particules fines sur la semaine de prédiction"},
    {"name": "average_So_2", "label": "Moyenne Dioxyde de soufre :", "tooltip": "Valeur Moyenne du Dioxyde de soufre sur la semaine de prédiction"},
    {"name": "avg_pressure", "label": "Pression atmosphérique moyenne :", "tooltip": "Moyenne de la pression atmosphérique de la semaine de prédiction"},
    {"name": "avg_temperature_max", "label": "Moyenne Température Maximum :", "tooltip": "Moyenne de la température maximum de la semaine de prédiction"},
    {"name": "dept", "label": "Département : ", "tooltip": "Numéro du département"},
    {"name": "indice", "label": "Indice de la grippe :", "tooltip": "Valeur de la grippe de la semaine"},
    {"name": "week_cos", "label": "Cosinus de la semaine :", "tooltip": "Transformation de la semaine sous forme trigonométrique"},
    {"name": "year", "label": "Année :", "tooltip": "Année de la prédiction"}
]


# Extraire uniquement les noms des features pour éviter les erreurs
feature_names = [f["name"] for f in features]


# Route principale qui affiche la page HTML avec le formulaire
@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/predict")
def predict_page():
    return render_template("prediction.html", features=features, data=df.to_dict(orient="records"))

# API pour récupérer les valeurs du CSV selon l'index
@app.route("/get_data/<int:index>")
def get_data(index):
    if 0 <= index < len(df):
        return jsonify(df.iloc[index].to_dict())
    return jsonify({"error": "Index hors limites"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les valeurs du formulaire et les afficher
        input_data = {feature: request.form.get(feature) for feature in feature_names}
        print("Données reçues :", input_data)

        # Vérifier les features manquantes
        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
            print("Features manquantes :", missing_features)
            return jsonify({"error": f"Features manquantes : {missing_features}"})

        # Conversion en DataFrame et en float
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype(float)

        # Vérifier la forme des données avant prédiction
        print("Nombre de features envoyées après renommage :", input_df.shape[1])
        print("Features envoyées après renommage :", input_df.columns.tolist())

        if input_df.shape[1] != len(feature_names):
            return jsonify({"error": f"Nombre de features incorrect : {input_df.shape[1]} au lieu de {len(feature_names)}."})

        # Faire la prédiction avec le modèle
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True)
