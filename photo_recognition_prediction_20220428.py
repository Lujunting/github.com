from ast import increment_lineno
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
import itertools
import matplotlib.pyplot as plt
import warnings
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plotImages(images_arr):
    fig, axes = plt.subplots(1,3,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# confusion matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
        horizontalalignment="center",
        color="red" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# run for sound_recognition
# def sound_to_photo(speech_result):
#開啟training完之jason檔
with open('model_trained.json','r') as f: 
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_trained.h5')

#  deal withe the test data
test_path = 'C:/Users/User/Pictures/animals/test/'
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
.flow_from_directory(directory=test_path,target_size=(200,200),classes=['cat','dog','wolf'],batch_size=10)

test_imgs, test_labels = next(test_batches)
# plotImages(test_imgs)
print(test_labels)
predictions = loaded_model.predict(x=test_batches,verbose=0)
predict_result=np.round(predictions)
# plot cofuion matrix
cm = confusion_matrix(y_true=test_batches.classes,y_pred=np.argmax(predictions,axis=-1))
print(cm)
# plot confuse matrix
cm_plot_labels = ['cat','dog','wolf']
plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title='Confusion Matrix')
print(cm)

cls_list = ['cats', 'dogs','wolf']

    # sound recognition
    # data_order = 0
    # for result in predict_result:
    #     if speech_result == 0:
    #         if np.array_equal(result,[1, 0, 0]):
    #             plt.imshow(test_imgs[data_order])
    #             plt.show()
    #             break
    #     elif speech_result == 1:
    #         if np.array_equal(result,[0, 1, 0]):
    #             plt.imshow(test_imgs[data_order])
    #             plt.show()
    #             break
    #     elif speech_result== 2:
    #         if np.array_equal(result,[0, 0, 1]):
    #             plt.imshow(test_imgs[data_order])
    #             plt.show()
    #             break    
    #     data_order += 1 

# photo recognition
if __name__ == '__main__':
    # 利用輸入其中一種動物來索引對應之圖片
    request_data = input('Input your request data (cat/dog/wolf):')
    data_order = 0
    for result in predict_result:
        print(result)
        if request_data == 'cat':
            if np.array_equal(result,[1, 0, 0]):
                plt.imshow(test_imgs[data_order])
                plt.show()
                print(cls_list[0])
                break
        elif request_data == 'dog':
            if np.array_equal(result,[0, 1, 0]):
                plt.imshow(test_imgs[data_order])
                plt.show()
                print(cls_list[1])
                break
        elif request_data == 'wolf':
            if np.array_equal(result,[0, 0, 1]):
                plt.imshow(test_imgs[data_order])
                plt.show()
                print(cls_list[2])
                break    
        data_order += 1 # 依序搜尋完所有照片

   

