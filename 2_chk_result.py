from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import transform
import os


img_height=128
img_width=128
img_vector=3


model = tf.keras.models.load_model('model\\male_female_20230627.h5')


def get_predict(file):
    img = Image.open(file)
    img = img.resize((img_height, img_width))
    img_np = np.array(img)
   
#    img_np = (img_np - 0.0)/(255.0 - 0.0)
    img_np = img_np.reshape(1,img_height,img_width, img_vector)
  
    
    predict = np.round(model.predict(img_np), 2) * 100

    
    print(predict)   
    
    return predict



file = "img_result\\woo2.jpg"

pred = get_predict(file)

print(pred.shape)


resultMsg = f'female={pred[0][0]}%, male={pred[0][1]}%'


print(resultMsg)

'''
(x_data, y_data), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_test.shape)

x_test = (x_test - 0.0)/(255.0 - 0.0)
predicted_value = model.predict(x_test)

print(y_test, predicted_value)

print(y_test)
print(np.argmax(predicted_value, axis=-1))

'''