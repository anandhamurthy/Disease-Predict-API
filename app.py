from flask import Flask, jsonify, request
import werkzeug
import flask
from tensorflow.keras import models
import numpy as np
from PIL import Image

app = Flask(__name__)

def getRiceDetails(n):
    l=['Brown Spot', 'Healthy', 'Hispa' ,'Leaf Blast']
    response = {
        'status': 200,
        'message': 'OK',
        'name': l[n]
    }
    return jsonify(response)

def getMaizeDetails(n):
    l=['Common Rust', 'Gray Leaf Spot', 'Blight' ,'Healthy']
    response = {
        'status': 200,
        'message': 'OK',
        'name': l[n]
    }
    return jsonify(response)

@app.route('/rice/',methods=['GET','POST'])
def rice():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    test_image = Image.open(filename)
    test_image=test_image.resize((224,224))
    test_image = np.asarray(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    loaded_model = models.load_model('rice.h5',custom_objects=None, compile=True)
    result = loaded_model.predict(test_image)
    return getRiceDetails(np.argmax(result))

@app.route('/maize/',methods=['GET','POST'])
def maize():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    test_image = image.load_img(filename, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    loaded_model = models.load_model('maize.h5')
    result = loaded_model.predict(test_image)
    return getMaizeDetails(np.argmax(result))

if __name__ == '__main__':
    app.run(debug=True)
