from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        
        # Convert prediction to Python float for JSON serialization
        prediction = float(prediction[0])
        
        # Return prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

