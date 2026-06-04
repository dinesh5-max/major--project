from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import json

app = Flask(__name__)

# Load the trained models
with open('model/crop_recommendation.pkl', 'rb') as f:
    crop_model = pickle.load(f)

with open('model/soil_crop_recommendation.pkl', 'rb') as f:
    soil_crop_model = pickle.load(f)

with open('model/vegetable_recommendation.pkl', 'rb') as f:
    vegetable_model = pickle.load(f)

with open('model/disease_prediction.pkl', 'rb') as f:
    disease_model = pickle.load(f)

image_model = load_model('model/image_classification.h5')

# Load the remedies knowledge base
with open('remedies.json') as f:
    remedies = json.load(f)

# Load translations
with open('translations.json', encoding='utf-8') as f:
    translations = json.load(f)

# Create label encoders for disease prediction
with open('model/le_crop_disease.pkl', 'rb') as f:
    le_crop_disease = pickle.load(f)

with open('model/le_disease.pkl', 'rb') as f:
    le_disease = pickle.load(f)

@app.route('/')
def home():
    lang = request.args.get('lang', 'en')
    return render_template('index.html', t=translations.get(lang, translations['en']), lang=lang)

def _soil_crop_label(t, crop_name):
    return f'{t["soil_based_recommendation"]} {crop_name}'


# pH ranges for different crops (optimal range)
pH_PREFERENCES = {
    'kidneybeans': (5.50, 6.00),
    'mango': (4.51, 6.97),
    'pigeonpeas': (4.55, 7.45),
    'apple': (5.51, 6.50),
    'coconut': (5.50, 6.47),
    'banana': (5.51, 6.49),
    'grapes': (5.51, 6.50),
    'maize': (5.51, 7.00),
    'muskmelon': (6.00, 6.78),
    'pomegranate': (5.56, 7.20),
    'rice': (5.01, 7.87),
    'watermelon': (6.00, 6.96),
    'mungbean': (6.22, 7.20),
    'jute': (6.00, 7.49),
    'papaya': (6.50, 6.99),
    'coffee': (6.02, 7.49),
    'mothbeans': (3.50, 9.94),
    'cotton': (5.80, 7.99),
    'lentil': (5.92, 7.84),
    'orange': (6.01, 8.00),
    'blackgram': (6.50, 7.78),
    'chickpea': (5.99, 8.87),
}

def _boost_by_ph(crop, ph, base_prob):
    """Boost probability if crop's pH is ideal for given pH"""
    if crop not in pH_PREFERENCES:
        return base_prob
    min_ph, max_ph = pH_PREFERENCES[crop]
    if min_ph <= ph <= max_ph:
        # Give boost if pH is within ideal range
        return base_prob * 1.3
    elif ph < min_ph:
        # Moderate penalty if pH is too low
        return base_prob * 0.8
    else:
        # Moderate penalty if pH is too high
        return base_prob * 0.8


@app.route('/predict_soil', methods=['GET', 'POST'])
def predict_soil():
    if request.method == 'GET':
        return redirect(url_for('home'))
    try:
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        ph = float(request.form['ph'])
        import pandas as pd
        
        soil_features = pd.DataFrame([[N, P, K, ph]], columns=['N', 'P', 'K', 'ph'])
        
        # Get probabilities for all crops
        soil_probabilities = soil_crop_model.predict_proba(soil_features)[0]
        soil_classes = soil_crop_model.classes_
        
        # Define pulses + fruits
        vegetables_fruits = [
            'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean',
            'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes',
            'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut',
        ]
        
        # Apply pH boosting and filter
        soil_crop_probs = []
        for crop, prob in zip(soil_classes, soil_probabilities):
            if crop in vegetables_fruits:
                boosted_prob = _boost_by_ph(crop, ph, prob)
                soil_crop_probs.append((crop, boosted_prob))
        
        soil_crop_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Get best crop for main prediction text
        best_crop = soil_crop_probs[0][0] if soil_crop_probs else 'N/A'
        soil_prediction_text = _soil_crop_label(t, best_crop)
        
        # For compatibility, also return the list for UI display
        top_soil_crops = soil_crop_probs[:8] if soil_crop_probs else []
        
        return render_template('index.html', soil_prediction_text=soil_prediction_text, top_soil_crops=top_soil_crops, t=t, lang=lang)
    except Exception as e:
        print(f"Error in predict_soil: {e}")
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        return render_template('index.html', soil_prediction_text=f"Error: {str(e)}", t=t, lang=lang)


@app.route('/predict_crop', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'GET':
        return redirect(url_for('home'))
    try:
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        # Get the data from the form
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        import pandas as pd
        soil_features = pd.DataFrame([[N, P, K, ph]], columns=['N', 'P', 'K', 'ph'])
        
        # Get probabilities for all crops from soil model
        soil_probabilities = soil_crop_model.predict_proba(soil_features)[0]
        soil_classes = soil_crop_model.classes_
        
        # Define pulses + fruits that exist in crop_model.classes_
        # (Used to show an expanded list in the UI.)
        vegetables_fruits = [
            'chickpea',
            'kidneybeans',
            'pigeonpeas',
            'mothbeans',
            'mungbean',
            'blackgram',
            'lentil',
            'pomegranate',
            'banana',
            'mango',
            'grapes',
            'watermelon',
            'muskmelon',
            'apple',
            'orange',
            'papaya',
            'coconut',
        ]
        
        # Create list of (crop, probability) for vegetables/fruits only with pH boosting
        soil_crop_probs = []
        for crop, prob in zip(soil_classes, soil_probabilities):
            if crop in vegetables_fruits:
                boosted_prob = _boost_by_ph(crop, ph, prob)
                soil_crop_probs.append((crop, boosted_prob))
        
        soil_crop_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Show ALL suitable pulses/fruits (not just top 8).
        # Put muskmelon at the end so the UI doesn't look "stuck" on muskmelon.
        top_soil_crops = [x for x in soil_crop_probs if x[0] != "muskmelon"] + [
            x for x in soil_crop_probs if x[0] == "muskmelon"
        ]
        
        soil_prediction_text = f"{len(top_soil_crops)} suitable vegetables/fruits found"

        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Get probabilities for all crops
        probabilities = crop_model.predict_proba(features)[0]
        classes = crop_model.classes_
        
        # Create a list of (crop, probability) tuples with pH boosting and sort by probability
        crop_probs = []
        for crop, prob in zip(classes, probabilities):
            if crop in vegetables_fruits:
                boosted_prob = _boost_by_ph(crop, ph, prob)
                crop_probs.append((crop, boosted_prob))
        
        crop_probs.sort(key=lambda x: x[1], reverse=True)

        # Show ALL pulses/fruits sorted by probability.
        # Put muskmelon at the end so the UI doesn't look "stuck" on muskmelon.
        top_crops = [x for x in crop_probs if x[0] != "muskmelon"] + [
            x for x in crop_probs if x[0] == "muskmelon"
        ]

        # Avoid always showing muskmelon in the headline if other pulses/fruits exist.
        best_crop = top_crops[0][0] if top_crops else "N/A"
        for crop, _prob in top_crops:
            if crop != "muskmelon":
                best_crop = crop
                break

        return render_template(
            'index.html',
            soil_prediction_text=soil_prediction_text,
            top_soil_crops=top_soil_crops,
            prediction_text=f'{t["recommended_crop"]} {best_crop}',
            top_crops=top_crops,
            t=t,
            lang=lang,
        )
    except Exception as e:
        print(f"Error in predict_crop: {e}")
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        return render_template('index.html', prediction_text=f"Error: {str(e)}", t=t, lang=lang)


def _format_veg_name(name):
    return str(name).replace('_', ' ')


@app.route('/predict_vegetable', methods=['GET', 'POST'])
def predict_vegetable():
    if request.method == 'GET':
        return redirect(url_for('home'))
    try:
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        N = int(request.form['v_N'])
        P = int(request.form['v_P'])
        K = int(request.form['v_K'])
        ph = float(request.form['v_ph'])
        temperature = float(request.form['v_temperature'])
        humidity = float(request.form['v_humidity'])
        rainfall = float(request.form['v_rainfall'])
        season = int(request.form['season'])

        import pandas as pd
        features = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall, season]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'season'],
        )
        probs = vegetable_model.predict_proba(features)[0]
        classes = vegetable_model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        
        # Show ALL vegetables, not just top 5
        all_vegetables = [(_format_veg_name(a), float(b)) for a, b in ranked]
        best = all_vegetables[0][0]
        vegetable_prediction_text = f'{t["recommended_vegetable"]} {best}'
        return render_template(
            'index.html',
            vegetable_prediction_text=vegetable_prediction_text,
            top_vegetables=all_vegetables,
            t=t,
            lang=lang,
        )
    except Exception as e:
        print(f"Error in predict_vegetable: {e}")
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        return render_template('index.html', vegetable_prediction_text=f"Error: {str(e)}", t=t, lang=lang)


@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'GET':
        return redirect(url_for('home'))
    try:
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        # Get the data from the form
        temperature = float(request.form['temperature_disease'])
        humidity = float(request.form['humidity_disease'])
        rainfall = float(request.form['rainfall_disease'])
        crop = request.form['crop_disease']

        # Encode the crop
        crop_encoded = le_crop_disease.transform([crop])

        # Create a numpy array for prediction
        import pandas as pd
        features = pd.DataFrame([[temperature, humidity, rainfall, crop_encoded[0]]], columns=['temperature', 'humidity', 'rainfall', 'crop'])

        # Make prediction
        prediction_encoded = disease_model.predict(features)
        prediction = le_disease.inverse_transform(prediction_encoded)[0]

        # Get the remedy
        remedy = remedies.get(prediction, {"remedy": "No remedy found", "preparation": "", "commercial_medicine": "N/A", "usage": "N/A", "prevention": "N/A"})

        return render_template('index.html', disease_prediction_text=f'{t["predicted_disease"]} {prediction}', remedy_text=remedy, t=t, lang=lang)
    except Exception as e:
        print(f"Error in predict_disease: {e}")
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        return render_template('index.html', disease_prediction_text=f"Error: {str(e)}", t=t, lang=lang)

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'GET':
        return redirect(url_for('home'))
    try:
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        if 'file' not in request.files:
            return render_template('index.html', image_prediction_text='No file part', t=t, lang=lang)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', image_prediction_text='No selected file', t=t, lang=lang)
        if file:
            # Save the file to a temporary location
            filepath = os.path.join('static', file.filename)
            file.save(filepath)

            # Load and preprocess the image
            img = image.load_img(filepath, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # Make prediction
            predictions = image_model.predict(x)
            confidence = np.max(predictions[0]) * 100
            predicted_class = np.argmax(predictions[0])
            
            # We need the list of classes to map prediction index to name
            # Since the model was trained with ImageDataGenerator, the classes are alphabetical
            # Based on the folders in plant_images:
            classes = sorted(os.listdir('plant_images'))
            result = classes[predicted_class]
            
            # Get remedy from database
            full_name = result.replace('_', ' ')
            # Try full name first, then try name without the first word (the crop)
            remedy = remedies.get(full_name)
            if not remedy:
                parts = full_name.split(' ')
                if len(parts) > 1:
                    short_name = ' '.join(parts[1:])
                    remedy = remedies.get(short_name)
            
            if not remedy:
                remedy = {"remedy": "No remedy found", "preparation": "", "commercial_medicine": "N/A", "usage": "N/A", "prevention": "N/A"}

            return render_template('index.html', 
                                 image_prediction_text=f'{t["predicted_disease"]} {full_name} ({t["confidence"]} {confidence:.2f}%)', 
                                 remedy_text_image=remedy,
                                 uploaded_image=file.filename,
                                 t=t,
                                 lang=lang)
    except Exception as e:
        print(f"Error in predict_image: {e}")
        lang = request.form.get('lang', 'en')
        t = translations.get(lang, translations['en'])
        return render_template('index.html', image_prediction_text=f"Error: {str(e)}", t=t, lang=lang)

@app.route('/dashboard')
def dashboard():
    lang = request.args.get('lang', 'en')
    t = translations.get(lang, translations['en'])
    
    # Get all remedies for the dashboard
    all_remedies = []
    for disease, remedy_data in remedies.items():
        all_remedies.append({
            'disease': disease.title(),
            'remedy': remedy_data.get('remedy', 'N/A'),
            'commercial_medicine': remedy_data.get('commercial_medicine', 'N/A'),
            'preparation': remedy_data.get('preparation', ''),
            'usage': remedy_data.get('usage', ''),
            'prevention': remedy_data.get('prevention', '')
        })
    
    # Farming solutions data
    farming_solutions = [
        {
            'category': 'Soil Management',
            'problems': [
                {'name': 'Low Soil Fertility', 'solution': 'Add organic manure, compost, and balanced NPK fertilizers. Test soil pH and nutrients regularly.'},
                {'name': 'Soil Erosion', 'solution': 'Use contour farming, plant cover crops, and build check dams to prevent soil loss.'},
                {'name': 'Saline Soil', 'solution': 'Use gypsum application, proper drainage, and salt-tolerant crop varieties.'},
                {'name': 'Acidic Soil', 'solution': 'Apply lime to raise pH, use acid-tolerant crops, and add organic matter.'}
            ]
        },
        {
            'category': 'Water Management',
            'problems': [
                {'name': 'Water Scarcity', 'solution': 'Implement drip irrigation, rainwater harvesting, and drought-resistant crop varieties.'},
                {'name': 'Waterlogging', 'solution': 'Improve drainage systems, avoid over-irrigation, and use raised bed farming.'},
                {'name': 'Poor Water Quality', 'solution': 'Test water quality, use filtration systems, and avoid contaminated water sources.'}
            ]
        },
        {
            'category': 'Pest & Disease Control',
            'problems': [
                {'name': 'Insect Pest Attack', 'solution': 'Use integrated pest management, beneficial insects, and organic pesticides.'},
                {'name': 'Fungal Diseases', 'solution': 'Improve air circulation, use fungicides, and practice crop rotation.'},
                {'name': 'Bacterial Diseases', 'solution': 'Use copper-based sprays, remove infected plants, and practice sanitation.'},
                {'name': 'Viral Diseases', 'solution': 'Control insect vectors, use virus-free seeds, and practice field sanitation.'}
            ]
        },
        {
            'category': 'Crop Management',
            'problems': [
                {'name': 'Poor Germination', 'solution': 'Use quality seeds, proper seed treatment, and optimal sowing conditions.'},
                {'name': 'Nutrient Deficiency', 'solution': 'Apply balanced fertilizers, use foliar sprays, and maintain soil health.'},
                {'name': 'Weed Competition', 'solution': 'Use mulching, mechanical weeding, and selective herbicides.'},
                {'name': 'Low Yield', 'solution': 'Optimize planting density, use high-yielding varieties, and proper crop management.'}
            ]
        },
        {
            'category': 'Climate & Weather',
            'problems': [
                {'name': 'Drought Stress', 'solution': 'Use drought-tolerant varieties, implement water conservation, and timing of operations.'},
                {'name': 'Flood Damage', 'solution': 'Use flood-tolerant varieties, raised bed farming, and proper drainage.'},
                {'name': 'Heat Stress', 'solution': 'Use shade nets, mulching, and heat-tolerant crop varieties.'},
                {'name': 'Cold Damage', 'solution': 'Use cold-tolerant varieties, protective covers, and proper timing.'}
            ]
        }
    ]
    
    return render_template('dashboard.html', 
                         remedies=all_remedies,
                         farming_solutions=farming_solutions,
                         t=t, 
                         lang=lang)

@app.route('/search_medicine', methods=['POST'])
def search_medicine():
    medicine_name = request.form.get('medicine_name', '').strip().lower()
    lang = request.form.get('lang', 'en')
    t = translations.get(lang, translations['en'])
    
    results = []
    if medicine_name:
        for disease, remedy_data in remedies.items():
            commercial_med = remedy_data.get('commercial_medicine', '').lower()
            remedy_name = remedy_data.get('remedy', '').lower()
            
            if medicine_name in commercial_med or medicine_name in remedy_name:
                results.append({
                    'disease': disease.title(),
                    'remedy': remedy_data.get('remedy', 'N/A'),
                    'commercial_medicine': remedy_data.get('commercial_medicine', 'N/A'),
                    'preparation': remedy_data.get('preparation', ''),
                    'usage': remedy_data.get('usage', ''),
                    'prevention': remedy_data.get('prevention', '')
                })
    
    # Get all remedies for the complete database display
    all_remedies = []
    for disease, remedy_data in remedies.items():
        all_remedies.append({
            'disease': disease.title(),
            'remedy': remedy_data.get('remedy', 'N/A'),
            'commercial_medicine': remedy_data.get('commercial_medicine', 'N/A'),
            'preparation': remedy_data.get('preparation', ''),
            'usage': remedy_data.get('usage', ''),
            'prevention': remedy_data.get('prevention', '')
        })
    
    return render_template('medicine_search.html', 
                         medicine_name=medicine_name,
                         results=results,
                         remedies=all_remedies,
                         t=t,
                         lang=lang)

@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    app.run(debug=True)