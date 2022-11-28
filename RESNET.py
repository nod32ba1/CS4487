from glob import glob # For Reading Data
import numpy as np # Helps in storing large data in NP Arrays
import time
import cv2 # For Image Processing
import sklearn # For Machine Learning
# import keras # For CNN
import tensorflow as tf # For CNN
from tensorflow import keras # For CNN
import matplotlib.pyplot as plt # For Data Visualization
from sklearn.model_selection import train_test_split # For splitting the data into training and testing set
from tensorflow.keras.models import Sequential # For CNN
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization # For adding layers to CNN
from tensorflow.keras.optimizers import Adam # For Learning Rate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # For Saving Model and Stopping Early if needed
from csv import DictReader # For Reading the CSV
from tensorflow.keras.applications import ResNet50 # For using ResNet-50 Model

def plotPerformance(hist,title):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.savefig(title)
    plt.show()

def getData():
    real = glob("../data/original/*")
    fake_all = glob("../data/manipulated/*")
    trainX = []
    trainY = np.hstack((np.ones(len(real), dtype=int),np.zeros(len(fake_all),dtype=int)))
    l = len(real+fake_all)
    for img_path in (real+fake_all):
        trainX.append(np.array(cv2.resize(cv2.imread(img_path),(150,150))) / 255.0)
        if(len(trainX)%(0.1*l)==0):print(f'process: {100*len(trainX)/l}%**')
    return np.asarray(trainX), trainY

# with zipfile.ZipFile("data.zip", 'r') as zip_ref:
#     zip_ref.extractall()
all_X,all_Y = getData()
trainX, testX, trainY, testY = train_test_split(all_X,all_Y,test_size=0.2, random_state=4487)

ResNet_model = Sequential()

ResNet_model.add(ResNet50(weights='imagenet', include_top=False, classes=2, input_shape=(150,150,3))) # Add a ResNet layer
ResNet_model.add(BatchNormalization()) # Normalize the input with mean close to 0 and standart deviation close to 1

ResNet_model.add(MaxPooling2D()) # Downsample the input by taking maximum value
ResNet_model.add(Flatten()) # Flatten the input

ResNet_model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform')) # output = activation(dot(input, kernel) + bias)
ResNet_model.add(Dropout(0.2)) # To prevent overfitting
ResNet_model.add(Dense(1, activation = 'sigmoid')) # output = activation(dot(input, kernel) + bias)

ResNet_model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=3e-4), metrics=['accuracy']) # Compile the model using the specified loss function and learning rate using accuracy score as the evaluation metric.

early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, verbose = 0, mode = 'min', restore_best_weights=True) # Stop early if the val_loss does not reduce for 5 epochs

checkpoint = ModelCheckpoint("ResNet.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto') # Save the best model with respect to val_accuracy

ResNet_hist=ResNet_model.fit(trainX,trainY,epochs=50,batch_size = 32,validation_data=(testX,testY), callbacks=[checkpoint, early_stopping]) # Fit the model


plotPerformance(ResNet_hist,"RESNET LR=3e-4 BS=32")

