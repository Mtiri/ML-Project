from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, optimizers, layers


vgg16_conv = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))

model = load_model('my_model001.h5')

new_model = Sequential()
new_model.add(vgg16_conv)
new_model.add(layers.Flatten())
new_model.add(model)

new_model.summary()

Classes = ["BALDE EAGLE", "EMPEROR PENGUIN", "SNOW OWL"]
# The local path to our target image
img_path = r'C:\Users\04710850298974836396\Pictures\test\eagle.jpg'
img1_path = r'C:\Users\04710850298974836396\Pictures\test\owl.jpg'
img2_path = r'C:\Users\04710850298974836396\Pictures\test\penguin.jpg'


# `img` is a PIL image of size 224x224
img00 = image.load_img(img_path, target_size=(224, 224))
img01 = image.load_img(img1_path, target_size=(224, 224))
img02 = image.load_img(img2_path, target_size=(224, 224))


# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img00)
x01 = image.img_to_array(img01)
x02 = image.img_to_array(img02)

#print('shape of array ',x.shape)


# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
x01 = np.expand_dims(x01, axis=0)
x02 = np.expand_dims(x02, axis=0)

#print('shape of array ',x.shape)
#x = np.reshape(x,(1, 244 * 244 * 3))
#img /= 255.


# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)
x01 = preprocess_input(x01)
x02 = preprocess_input(x02)


preds = new_model.predict(x)
preds01 = new_model.predict(x01)
preds02 = new_model.predict(x02)

print('eagle img class Prediction:', (preds))
print('Owl img class Prediction:', (preds01))
print('Penguin img class Prediction:', (preds02))
#classes labels
print("class prediction for eagle img is :", Classes[np.argmax(preds)])
print("class prediction for Owl img is :", Classes[np.argmax(preds01)])
print("class prediction for penguin img is :", Classes[np.argmax(preds02)])

