from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Convolution2D,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale = 1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test = ImageDataGenerator(rescale = 1./255)

A_train = train.flow_from_directory('/content/drive/MyDrive/Dataset/train', target_size=(64,64), color_mode='grayscale',batch_size=3, class_mode='categorical')
A_test = test.flow_from_directory('/content/drive/MyDrive/Dataset/test', target_size=(64,64), color_mode='grayscale',batch_size=3, class_mode='categorical')

print(A_train.class_indices)

print(A_test.class_indices)

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=512,activation='relu'))

model.add(Dense(units=6,activation='softmax'))

model.summary()

model.compile(metrics=['accuracy'],loss='categorical_crossentropy',optimizer='adam')
model.fit(A_train,steps_per_epoch = 594/3,epochs=25,validation_data=A_test,validation_steps=len(A_test))
model.save('gesture.h5')
json_model = model.to_json()
with open("model-gesture.json","w") as json_file:
  json_file.write(json_model)


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
test_model = load_model('gesture.h5')
img_path="/content/drive/MyDrive/test_image.jpg"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()

imgload = image.load_img(img_path,color_mode='grayscale',target_size=(64,64))
res = image.img_to_array(imgload)
res.shape
type(res)
res = np.expand_dims(res,axis=0)
res.shape
pred_res = np.argmax(test_model.predict(res),axis=-1)
pred_res
index = ['0','1','2','3','4','5']
final_res = str(index[pred_res[0]])
final_res
