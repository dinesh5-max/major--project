import pickle
import pandas as pd


with open("model/crop_recommendation.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("model/soil_crop_recommendation.pkl", "rb") as f:
    soil_crop_model = pickle.load(f)

vegetables_fruits = [
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
]

N, P, K, ph = 90, 42, 43, 6.5
temperature, humidity, rainfall = 25, 75, 100

soil_features = pd.DataFrame([[N, P, K, ph]], columns=["N", "P", "K", "ph"])
soil_probs = soil_crop_model.predict_proba(soil_features)[0]
soil_classes = soil_crop_model.classes_
soil_filtered = [(c, p) for c, p in zip(soil_classes, soil_probs) if c in vegetables_fruits]
soil_filtered.sort(key=lambda x: x[1], reverse=True)

features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
probs = crop_model.predict_proba(features)[0]
classes = crop_model.classes_
full_filtered = [(c, p) for c, p in zip(classes, probs) if c in vegetables_fruits]
full_filtered.sort(key=lambda x: x[1], reverse=True)

print("Soil filtered count:", len(soil_filtered))
print("Top 8 soil pulses/fruits:", soil_filtered[:8])
print("Full filtered count:", len(full_filtered))
print("Top 8 full pulses/fruits:", full_filtered[:8])

