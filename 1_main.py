import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import transform
import os
import matplotlib.pyplot as plt


img_height=250
img_width=200

batch_size = 200

dir_img_basic = os.path.join(os.getcwd(), "img_dataset")
dir_img_train = os.path.join(dir_img_basic, "train")
dir_img_test = os.path.join(dir_img_basic, "test")

down_data_class=[
    'female',
    'male'
]

directory_list = [
    dir_img_train,
    dir_img_test,
]

def loadImg(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')
   np_image = transform.resize(np_image, (img_height, img_width, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image



def loss_graph(hist):
   plt.plot(hist.history['loss'])
   plt.plot(hist.history['val_loss'])
   plt.xlabel('epoch')
   plt.ylabel('loss')
   plt.legend(['train', 'val'])
   plt.show()


def my_model():
    x = tf.keras.layers.Input(shape=[img_height, img_width, 3])
    
    h = tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height,img_width,3))(x)
    h = tf.keras.layers.RandomRotation(0.1)(h)
    h = tf.keras.layers.RandomZoom(0.1)(h)
    h = tf.keras.layers.RandomBrightness(0.1)(h)   
    

    h = tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))(h)    

    h = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='swish', padding='same')(h)
    h = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='swish', padding='same')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(h)
    h = tf.keras.layers.Dropout(0.25)(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='swish', padding='same')(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='swish', padding='same')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(h)
    h = tf.keras.layers.Dropout(0.25)(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(h)
    h = tf.keras.layers.Dropout(0.25)(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(h)
    h = tf.keras.layers.Dropout(0.25)(h)
    h = tf.keras.layers.BatchNormalization()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(h)
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(h)
    h = tf.keras.layers.Dropout(0.25)(h)

    h = tf.keras.layers.Flatten()(h)

    h = tf.keras.layers.Dense(128, activation='relu')(h)
    h = tf.keras.layers.Dropout(0.5)(h)
    y = tf.keras.layers.Dense(2, activation='softmax')(h)

    model = tf.keras.models.Model(x,y)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics='accuracy')

    return model


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir_img_train,
    label_mode= "categorical", # int, binary
    color_mode='rgb',
    batch_size=batch_size,
    image_size = (img_height, img_width), # reshape if not in this size
    shuffle=True,
    seed = 123,
    validation_split=0.2,
    subset="training",
)


val_ds  = tf.keras.preprocessing.image_dataset_from_directory(
    dir_img_train,
    label_mode= "categorical", # int, binary
    color_mode='rgb',
    batch_size=batch_size,
    image_size = (img_height, img_width), # reshape if not in this size
    shuffle=True,
    seed = 123,
    validation_split=0.2,
    subset="validation",

)

model = my_model()
history = model.fit(train_ds, epochs=1, verbose=1, validation_data=val_ds)

loss_graph(history)

model.save("model\\male_femal_20230627.h5")

# Check result dataset
for dclass in down_data_class:
  img_path = dir_img_test + dclass

  for filename in os.listdir(img_path):
    full_file = img_path + "/" + filename

    np_image = loadImg(full_file)
    print(full_file, down_data_class[int(model.predict(np_image).argmax())])
    print(model.predict(np_image))
