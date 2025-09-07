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

@app.route('/new/')
def new_home_page():
    return render_template('html/index.html')
    

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
