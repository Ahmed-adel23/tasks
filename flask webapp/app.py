from flask import Flask, request, render_template, jsonify, send_file
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load model
model = load_model('cnn_model.h5')

# Set the expected input shape for the model
input_shape = (150, 150, 3)

# Function to preprocess and predict
def preprocess_and_predict(image_path):
    try:
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image / 255.0  # Normalize the image

       
        prediction = model.predict(np.expand_dims(image, axis=0))

      
        class_label = np.argmax(prediction)

        
        class_name = get_code(class_label)

        
        accuracy = float(np.max(prediction))

        return class_name, accuracy, image

    except Exception as e:
        return str(e), None, None

# Function to map class label to class name
def get_code(n):
    code = {'Control-Axial': 0, 'Control-Sagittal': 1, 'MS-Axial': 2, 'MS-Sagittal': 3}
    for x, y in code.items():
        if n == y:
            return x

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
      
        image = request.files['image']

        
        image_path = os.path.join('static', 'temp_image.jpg')
        image.save(image_path)


        predicted_class, accuracy, uploaded_image = preprocess_and_predict(image_path)

        if predicted_class is not None:
            # Return the predicted class label, confidence, and the uploaded image
            return render_template('index.html', class_label=predicted_class, accuracy= accuracy , image_path=image_path)
        else:
            return 'Error processing the image.'

    except Exception as e:
        return str(e)

@app.route('/image/<filename>')
def get_image(filename):
    # Serve the uploaded image
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

