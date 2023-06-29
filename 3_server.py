from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request
import io
from datetime import datetime

img_height=128
img_width=128
img_vector=3

app = Flask(__name__)
model = tf.keras.models.load_model('model\\male_female_20230627.h5')



def get_predict(file):
    img = Image.open(io.BytesIO(file))

    img = img.resize((img_height, img_width))    

    now = datetime.now()    
    save_file_name = str(now.date()) + "_" + str(now.time().hour) + str(now.time().minute) + str(now.time().second)

    img.save("img_req\\"+save_file_name + ".jpg")    
    
    img_np = np.array(img)    

    img_np = img_np.reshape(1,img_height,img_width, img_vector)

    predict = np.round(model.predict(img_np), 2) * 100
   
    print(save_file_name, predict)   
    
    return predict


@app.route('/', methods=['POST'])
def hello():    
    try:
        file = request.files['file']
        image_bytes = file.read()

        pred = get_predict(image_bytes)
        resultMsg = f'female={pred[0][0]} %,  male={pred[0][1]} %\n'
        return resultMsg
    except Exception as ex:
        print(ex)
        return "Can't read file from body"


app.run(host="0.0.0.0",port=8092)
