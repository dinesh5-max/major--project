# AI-Based Crop Recommendation and Health Monitoring System (Kisan Sahayak)

This project is an AI-powered web application designed to assist farmers in making informed decisions about crop selection, disease prediction, and treatment.

## Features

- **Crop Recommendation:** Suggests the best crop to grow based on soil parameters (Nitrogen, Phosphorus, Potassium, pH) and weather conditions (temperature, humidity, rainfall).
- **Disease & Pest Prediction:** Predicts potential diseases and pests based on weather data.
- **Image-based Health Analysis:** Analyzes images of plant leaves and roots to detect diseases using a Convolutional Neural Network (CNN).
- **Holistic Treatment:** Provides both Ayurvedic/natural remedies and commercial chemical solutions for identified issues.

## Technologies Used

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn, TensorFlow, Keras
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Data Handling:** Pandas, NumPy

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 installed. Then, run the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will be accessible at `http://127.0.0.1:5000`.

## Project Architecture

The system follows a modular architecture where the Flask backend integrates multiple Machine Learning and Deep Learning models.

```mermaid
graph TD
    A[User Input] --> B{Input Type}
    B -- Soil/Weather Data --> C[Random Forest Model]
    B -- Image Upload --> D[CNN Image Model]
    
    C -- Prediction --> E[Result Processor]
    D -- Prediction --> E
    
    E --> F[Knowledge Base (remedies.json)]
    F -- Fetch Treatment --> G[Ayurvedic Remedies]
    F -- Fetch Treatment --> H[Commercial Medicines]
    F -- Fetch Treatment --> I[Preventive Measures]
    
    G & H & I --> J[Dashboard Display]
```

## Project Structure

- `app.py`: The main Flask application file.
- `model/`: Contains all the trained machine learning models (`.pkl` and `.h5` files).
- `templates/`: Contains the HTML template (`index.html`).
- `static/`: Contains static files like CSS and JavaScript.
- `*.csv`: Datasets used for training the models.
- `*.ipynb`: Jupyter notebooks used for model training and experimentation.
- `remedies.json`: A knowledge base of treatments for various diseases and pests.
