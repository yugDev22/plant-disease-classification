import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.models import model_from_json
application = app = Flask(__name__)
potato_labels = ['Early blight','Late blight','Healthy','Bacterial','Black heart','Black scurf','Common scab','Septoria leaf']
tomato_labels = ['Bacterial spot','Early blight','Late blight','Leaf mold','Septoria leaf spot','Spider mites two-spotted spider mite','Target spot','Yellow leaf curl virus','Mosaic virus','Healthy']
wheat_labels = ['Healthy','Leaf rust','Stem rust']
rice_labels = ['Bacterial leaf blight','Brown spot','Leaf smut']
guava_labels = ['Canker','Dot','Mummification','Rust']
path=os.path.dirname(__file__)
TARGET_SIZE=(64,64)
def predict_disease(img_path,label):
  model_dir = path
  if label == "tomato":
    model_json_path = model_dir+"/"+label+"_model.json"
    model_h5_path = model_dir+"/"+label+"_model.h5"
    disease_class_labels = tomato_labels
    img = image.load_img(img_path, target_size=(128,128))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5_path)
    p = model.predict(img/255.0)
    v = np.argmax(p)
  elif label == "potato":
    model_json_path = model_dir+"/"+label+"_model.json"
    model_h5_path = model_dir+"/"+label+"_model.h5"
    disease_class_labels = potato_labels
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5_path)
    p = model.predict(img/255.0)
    v = np.argmax(p)
  elif label == "wheat":
    model_json_path = model_dir+"/"+label+"_model.json"
    model_h5_path = model_dir+"/"+label+"_model.h5"
    disease_class_labels = tomato_labels
    disease_class_labels = wheat_labels
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5_path)
    p = model.predict(img/255.0)
    v = np.argmax(p)
  elif label == "rice":
    model_json_path = model_dir+"/"+label+"_model.json"
    model_h5_path = model_dir+"/"+label+"_model.h5"
    disease_class_labels = tomato_labels
    disease_class_labels = rice_labels
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5_path)
    p = model.predict(img/255.0)
    v = np.argmax(p)

  elif label == "guava":
    model_json_path = model_dir+"/"+label+"_model.json"
    model_h5_path = model_dir+"/"+label+"_model.h5"
    disease_class_labels = tomato_labels
    disease_class_labels = guava_labels
    img = image.load_img(img_path, target_size=TARGET_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5_path)
    p = model.predict(img/255.0)
    v = np.argmax(p)
  else:
    return "Invalid Entry"
  k = disease_class_labels[v]
  result = str(k)+" : "+str(round(p[0][v]*100,2))+" %"
  return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
  
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        crop=request.form.get('crop')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = predict_disease(file_path,crop)
        result=preds
        return "Your "+ str(crop)+ " is affected with "+str(result) 
    return None
if __name__ == "__main__":
    app.run(debug=True)
