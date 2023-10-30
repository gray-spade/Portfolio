#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
train_labels = np.load("/content/drive/MyDrive/전공 과제/딥러닝/Y.npy")
train_images = np.load("/content/drive/MyDrive/전공 과제/딥러닝/X.npy")
X_train,X_test,y_train,y_test= train_test_split(train_images,train_labels, test_size=0.1,random_state=1004)


# In[4]:


from keras.utils import np_utils
categori ={'please': 0, 'whats up': 1,
           'no': 2, '9': 3, '0': 4, '7': 5,
           '6': 6, '1': 7, 'bye': 8, '8': 9,
           'a': 10, 'project': 11, 'pardon': 12,
           'NULL': 13, 'good': 14, 'little bit': 15,
           'hello': 16, 'yes': 17, 'good morning': 18,
           'c': 19, 'd': 20, '4': 21, '3': 22, 'e': 23,
           'b': 24, '2': 25, '5': 26}
def toint_Y(array,categori):
  
  count=0
  labels_toint= []
  for i in range(len(array)):
    labels_toint.append([categori[array[i][0]]])
  return labels_toint
train_Y = toint_Y(y_train,categori)
test_Y = toint_Y(y_test,categori)


# In[5]:


incoded_train=np_utils.to_categorical(train_Y)
incoded_test=np_utils.to_categorical(test_Y)


# In[6]:


del train_labels
del train_images
del train_Y
del test_Y


# In[7]:


from tensorflow.keras import models, layers
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(layers.Dense(units=27, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


history = model.fit(X_train, incoded_train, epochs=10, batch_size=32,validation_data=(X_test, incoded_test),
                    verbose=2
)


# In[8]:


history_dict = history.history
loss_values = history_dict['loss']		# 훈련 데이터 손실값
val_loss_values = history_dict['val_loss']	# 검증 데이터 손실값
acc = history_dict['accuracy']			# 정확도
epochs = range(1, len(acc) + 1)		# 에포크 수
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Plot')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train error', 'val error'], loc='upper left')
plt.show()


# In[9]:


results = model.evaluate(X_test, incoded_test, verbose=2)
print(results)


# In[13]:


from tensorflow.keras.preprocessing import image
img_path="/content/a.jpg"
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img) 
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
from matplotlib import pyplot
pyplot.imshow(img)
plt.axis('off')
pyplot.show()
print("추정된 수화=", [k for k, v in categori.items() if v == pred.argmax()])

