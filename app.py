from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data.csv')

# Define the expected features
expected_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = df[expected_features + ['MEDV']]

# Split the data into training and testing sets
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure all expected features are present
        if not all(feature in data for feature in expected_features):
            return jsonify({'error': 'Missing one or more features'}), 400

        # Extract features in the correct order
        features = [data[feature] for feature in expected_features]

        # Make prediction
        prediction = model.predict([features])
        # Multiply the prediction by 1000
        prediction_value = prediction[0] * 1000 * 83

        return jsonify({'prediction': prediction_value})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
