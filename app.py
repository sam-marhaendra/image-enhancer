from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template, jsonify
from img_enhancement import *
import pyrebase
import os
import datetime

config = {
  "apiKey": "AIzaSyA6aoM-VBk0mjr4JZxmmTBzG87S0c4B3sI",
  "authDomain": "image-20187.firebaseapp.com",
  "databaseURL": "https://image-20187-default-rtdb.firebaseio.com/",
  "projectId": "image-20187",
  "storageBucket": "image-20187.appspot.com",
  "serviceAccount": "serviceAccountKey.json"
}

firebase_storage = pyrebase.initialize_app(config)
storage = firebase_storage.storage()
auth = firebase_storage.auth()

app = Flask(__name__)
app.secret_key = "r32qi0j3q20j"
UPLOAD_FOLDER = 'static/uploads/'
 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
     
@app.route('/image', methods=['POST'])
def result():
    img = request.json['image']
    res = image_enhancement(img)
    res = res * 255
    date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.jpg'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], date_string), res)
    storage.child(date_string).put(os.path.join(app.config['UPLOAD_FOLDER'])+date_string)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'])+date_string)
    email = 'syafiqma69@gmail.com'
    password = 'fireBasePass'
    user = auth.sign_in_with_email_and_password(email, password)
    url = storage.child(date_string).get_url(user['idToken'])
    return jsonify({'url':url})
 
if __name__ == "__main__":
    app.run()
