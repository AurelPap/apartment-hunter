from flask import Flask, render_template, request
import joblib

# Charger le modèle pré-entraîné
model = joblib.load('modele.pkl')

# Créer une application Flask
app = Flask(__name__)

# Définir la route pour votre application
@app.route('/')
def home():
    return render_template('index.html')

# Définir une fonction pour faire la prédiction
def predict(features):
    prediction = model.predict([features])
    return prediction[0]

# Définir une route pour gérer la prédiction
@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        features = [float(request.form['feature1']), float(request.form['feature2']), float(request.form['feature3']), 
                    float(request.form['feature4']), bool(request.form['feature5']), int(request.form['feature6']), 
                    int(request.form['feature7']), float(request.form['feature8']), float(request.form['feature9']), 
                    int(request.form['feature10'])]
        prediction = predict(features)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
    
