import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, optimizers, layers



# load model exluding fully-connected layers
vgg16_conv = vgg16.VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
vgg16_conv.summary()

#Prepare the training data
#each folder contains three subfolders in accordance with the number of classes

train_dir = r'C:\Users\04710850298974836396\Pictures\dataset\trainset'
validation_dir = r'C:\Users\04710850298974836396\Pictures\dataset\validationset'

#the number of images for train and test :
ntrain = 300
nval = 75

#load the normalized images and use data augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
#data aygmentation used only for trainig
valid_datagen = ImageDataGenerator(rescale=1./255)

#define the batch size
batch_size = 20

#the defined shape is equal to the network output tensor shape
train_features = np.zeros(shape=(ntrain, 7, 7, 512))
train_labels = np.zeros(shape=(ntrain, 3))

#generate batches of train images and labels
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(244, 244),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True )


#Obtain model predictions on the train data
#get the predictions passing the images from train images and labels

for i, (inputs_batch, labels_batch) in enumerate(train_generator):
    if i * batch_size >= ntrain:
        break
    #pass the images throught the network
    features_batch = vgg16_conv.predict(inputs_batch)
    train_features[i * batch_size : (i+1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch

# reshape train_features into vector
train_features_vec = np.reshape(train_features, (ntrain, 7 * 7 * 512))
print("Train features: {}".format(train_features_vec.shape))

#Prepare the validation data
validation_features = np.zeros(shape=(nval, 7, 7, 512))
validation_labels = np.zeros(shape=(nval,3))

#generate batches of validation images and labels
validation_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=(244,244),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

#Visualize the validation dataset
#choose the image index for the visualization
val_image_id = 0
# get the validation image shape
print("The shape of validation images: {}".format(validation_generator[val_image_id][0][0].shape))

#visualize the image example
plt.axis('off')
plt.imshow(validation_generator[val_image_id][0][0])

#get image class and map its index with the names of the classes
val_image_label_id = np.argmax(validation_generator[val_image_id][1][0])
classes_list = list(validation_generator.class_indices.keys())

#show image class
plt.title("Class name: {}".format(classes_list[val_image_label_id]))
plt.show()

#Obtain model predictions on the validation data
#iterate through the batches of validation images and labels

for i, (inputs_batch, labels_batch) in enumerate(validation_generator):
    if i * batch_size >= nval:
        break
    features_batch = vgg16_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i+1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch

#reshape validation_features into vector
validation_features_vec = np.reshape(validation_features,(nval, 7 * 7 * 512))
print("Validation features: {}".format(validation_features_vec.shape))

#Create our own model and train the network
#Configure and train the model

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(Dropout(0.7))
model.add(Dense(3, activation='softmax'))

model.summary()

#configure the model for training
model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                loss='categorical_crossentropy',
                metrics=['acc'])

#use the train and validation feature vectors
history = model.fit(train_features_vec,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features_vec,validation_labels))

#Show the errors
# get the list of all validation file names
fnames = validation_generator.filenames

# get the list of the corresponding classes
ground_truth = validation_generator.classes

# get the dictionary of classes
label2index = validation_generator.class_indices

# obtain the list of classes
idx2label = list(label2index.keys())
print("The list of classes: ", idx2label)


predictions = model.predict_classes(validation_features_vec)
prob = model.predict(validation_features_vec)

errors = np.where(predictions != ground_truth)[0]
print("Number of errors = {}/{}".format(len(errors),nval))


#Let's take a look at the loss and accuracy curves during training:

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]

    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))

    original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
    plt.axis('off')
    plt.imshow(original)
    plt.show()

# save the model
model.save('my_model001.h5')
