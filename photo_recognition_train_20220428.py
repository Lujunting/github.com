from grpc import Future
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.layers import Dropout
from kerastuner.engine.hyperparameters import HyperParameters
import pickle #tune完參數後會存成.pkl file
import itertools
import matplotlib.pyplot as plt
import warnings
import json
import time # 將訓練時間記錄下來
from tensorflow.keras.callbacks import TensorBoard
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
from keras import optimizers
from tensorflow.keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.layers import Input, Dense, SimpleRNN, RNN, LSTM


# 導入tensorboard追蹤訓練趨勢 ；f 指
NAME = f'photo_recog{int(time.time())}'     # time 會以數字編碼呈現
tensorboard = TensorBoard(log_dir=f'photo_training_data/{NAME}') # save as .log file

# zip 為打包為元组的列表，以陣列顯示
def plotImages(images_arr):
    fig, axes = plt.subplots(1,3,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# warnings 發出異常警告，ignore為忽略警告
warnings.simplefilter(action='ignore',category=FutureWarning)
train_path = 'C:/Users/User/Pictures/animals/train/'
valid_path = 'C:/Users/User/Pictures/animals/valid/'
test_path = 'C:/Users/User/Pictures/animals/test/'
        

# # 建立 train/valid/test batches
train_batches = ImageDataGenerator(rescale=1/255,preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory=train_path,target_size=(244,244),classes=['cat','dog','wolf'],batch_size=15)

valid_batches = ImageDataGenerator(rescale=1/255,preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory=valid_path,target_size=(244,244),classes=['cat','dog','wolf'],batch_size=15)

test_batches = ImageDataGenerator(rescale=1/255,preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory=test_path,target_size=(244,244),classes=['cat','dog','wolf'],batch_size=15)

imgs, labels = next(train_batches)
plotImages(imgs)
print(labels)

## training model (kernel_size 為捲積層大小；strides為滑動步長)
model = Sequential([
    Conv2D(filters=32, kernel_size=(5,5),activation='relu',padding='same',input_shape=(244,244,3)),
    MaxPool2D(pool_size=(2,2),strides=1),
    ## 加入 L2 正則化，0.01為正則化參數
    Dense( 32, input_dim=32,kernel_regularizer=regularizers.l2(0.01)),
    Conv2D(filters=32, kernel_size=(5,5),activation='relu',padding='same'),
    MaxPool2D(pool_size=(2,2),strides=1),
    Conv2D(filters=64, kernel_size=(5,5),activation='relu',padding='same'),
    MaxPool2D(pool_size=(2,2),strides=1),
    Dense(128,activation='relu'),
    Flatten(), # 3維資料特徵轉化為1維
    # Dropout(0.3),
    Dense(units=3,activation='softmax')	# unit=3: cat & dog & wolf
])

model.summary()

# model evaluation 
# sgd=optimizers.gradient_descent_v2.SGD(learning_rate=0.001,momentum=0.9,decay=1e-6)
model.compile(optimizer=Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.1),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x=train_batches,validation_data=valid_batches,epochs=15,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3
,min_delta=0.001)])

# 儲存訓練結果為jason檔
model_json = model.to_json()
with open("photo_model_trained.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save("photo_model_trained_test.h5") # 將模型儲存至 HDF5 檔案中
model.save_weights("photo_model_test.weight")

# 繪製訓練 & 驗證損失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plot_model(model, to_file='model_loss.png')

#繪製訓練 & 驗證準確值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



