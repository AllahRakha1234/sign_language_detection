from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2


class Model:

  classifier = None
  def __init__(self, Type):
    self.classifier = Type
    
  
  def build_model(classifier):
    

    classifier.add(Convolution2D(128, (3, 3), input_shape=(64, 64, 1), activation='relu'))

    classifier.add(Convolution2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    classifier.add(Convolution2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    classifier.add(Convolution2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    classifier.add(Flatten())

    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(1024, activation='relu'))
    

    classifier.add(Dense(29, activation='softmax'))

    return classifier

  def save_classifier(path, classifier):
    classifier.save(path)

  def load_classifier(path):
    classifier = load_model(path)
    return classifier

  def predict(classes, classifier, img):
    img = cv2.resize(img, (64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0

    pred = classifier.predict(img)
    return classes[np.argmax(pred)], pred
    

class DataGatherer:

  def __init__(self, *args):
    if len(args) > 0:
      self.dir = args[0]
    elif len(args) == 0:
      self.dir = ""


  #this function loads the images along with their labels and apply
  #pre-processing function on the images and finaly split them into train and
  #test dataset
  def load_images(self):


    # x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
     x_train = np.load('x_train.npy')
     x_test = np.load('x_test.npy')
     y_train = np.load('y_train.npy')
     y_test = np.load('y_test.npy')

        # return x_train, x_test, y_train, y_test

     return x_train, x_test, y_train, y_test

  def edge_detection(self, image):
    minValue = 70
    blur = cv2.GaussianBlur(image,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

