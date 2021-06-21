# Adapted from https://www.analyticsvidhya.com/blog/2020/12/deep-learning-with-google-teachable-machine/

# Import necessary modules
import numpy as np
import cv2
from time import sleep
import tensorflow.keras
from keras.preprocessing import image
import tensorflow as tf
import pyautogui

# Using laptop's webcam as source of video
cap = cv2.VideoCapture(0)

# Labels - The various possibilities
labels = ['Neutral','Up','Down','Left','Right']

# Loading the model weigths
model = tensorflow.keras.models.load_model('keras_model.h5')

while True:

   success, image = cap.read()

   if success == True:
      # Necessary to avoid conflict between left and right
      image = cv2.flip(image,1)
      cv2.imshow("Frame",image)

      #Take pic
      image_array = cv2.resize(image,(224,224))

      # Normalize the image
      normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

      # Load the image into the array
      data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
      data[0] = normalized_image_array

      # run the inference
      prediction = model.predict(data)
      print(prediction)

      # Map the prediction to a class name
      predicted_class = np.argmax(prediction[0], axis=-1)
      predicted_class_name = labels[predicted_class]

      # Using pyautogui to get the current position of the mouse and move accordingly
      current_pos = pyautogui.position()
      current_x = current_pos.x
      current_y = current_pos.y

      print(predicted_class_name)

      if predicted_class_name == 'Neutral':
         sleep(1)
      elif predicted_class_name == 'Left':
         pyautogui.moveTo(current_x-80,current_y,duration=1)
         sleep(1)
      elif predicted_class_name == 'Right':
         pyautogui.moveTo(current_x+80,current_y,duration=1)
         sleep(1)
      elif predicted_class_name == 'Down':
         pyautogui.moveTo(current_x,current_y+80,duration=1)
         sleep(1)
      elif predicted_class_name == 'Up':
         pyautogui.moveTo(current_x,current_y-80,duration=1)
         sleep(1)

# Release open connections
cap.release()
cv2.destroyAllWindows()	    	
