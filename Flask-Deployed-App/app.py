import os
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load data and model once on startup
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

model = CNN.CNN(39)  # Full model
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Mapping from plant type to class indices
plant_class_map = {
    'apple': [0, 1, 2, 3],
    'cherry': [6, 7],
    'corn': [8, 9, 10, 11],
    'grape': [12, 13, 14, 15],
    'orange': [16],
    'peach': [17, 18],
    'pepper': [19, 20],
    'potato': [21, 22, 23],
    'raspberry': [24],
    'soybean': [25],
    'squash': [26],
    'strawberry': [27, 28],
    'tomato': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    # Add or adjust more types as needed
}

# Inference function
def prediction(image_path, plant_type):
    if plant_type not in plant_class_map:
        raise ValueError(f"Unsupported plant type: {plant_type}")
    plant_type = plant_type.lower()
    valid_indices = plant_class_map[plant_type]
    
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    
    with torch.no_grad():
        output = model(input_data).numpy().flatten()

    filtered_output = output[valid_indices]
    pred_index_in_filtered = np.argmax(filtered_output)
    final_index = valid_indices[pred_index_in_filtered]

    return final_index

# Flask app
app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'plantType' not in request.form:
        return jsonify({'error': 'No plant type provided'}), 400

    plant_type = request.form['plantType'].lower()
    image = request.files['image']
    filename = image.filename
    file_path = os.path.join('uploads', filename)

    os.makedirs('uploads', exist_ok=True)
    image.save(file_path)

    try:
        pred = prediction(file_path, plant_type)

        result = {
            'diseaseName': disease_info['disease_name'][pred],
            'description': disease_info['description'][pred],
            'prevention': disease_info['Possible Steps'][pred],
            # 'diseaseImageUrl': disease_info['image_url'][pred],
            'supplementName': supplement_info['supplement name'][pred],
            # 'supplementImageUrl': supplement_info['supplement image'][pred],
            # 'supplementBuyLink': supplement_info['buy link'][pred],
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
