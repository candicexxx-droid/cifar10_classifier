import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  # output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  
  output_ims = (input_images - np.amin(input_images))/(np.amax(input_images) - np.amin(input_images))
  ran2 = 2
  output_ims = (output_ims*ran2) - 1;
  # normalization = tf.keras.layers.Normalization()
  # output_ims = tf.keras.layers.Normalization(input_images)
  return output_ims

def fetch_data(save_dir = "/content/drive/MyDrive/VCLA-research/cifar10_np_data/"):
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  print("data shape info: training , testing (image + label)")
  print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


  x_train = preprocess_image_input(x_train)
  x_test = preprocess_image_input(x_test)

  print(x_test[0])
  print("max_train: ",np.amax(x_train))
  print("min_train: ",np.amin(x_train))
  np.save(save_dir+"x_train.npy", x_train)
  np.save(save_dir+"y_train.npy", y_train)
  np.save(save_dir+"x_test.npy", x_test)
  np.save(save_dir+"y_test.npy", y_test)

  print("all data saved to " + save_dir)


fetch_data()