import tensorflow as tf
print("tf.__version__  ",tf.__version__) 
import numpy as np
import time
from model import define_compile_model

##load data

x_train = np.load("/content/drive/MyDrive/VCLA-research/cifar10_np_data/x_train.npy")
y_train = np.load("/content/drive/MyDrive/VCLA-research/cifar10_np_data/y_train.npy")
x_test = np.load("/content/drive/MyDrive/VCLA-research/cifar10_np_data/x_test.npy")
y_test = np.load("/content/drive/MyDrive/VCLA-research/cifar10_np_data/y_test.npy")

print("data load success!")
#Hyperparam:
EPOCHS = 4
BATCH_SIZE=64


model = define_compile_model()

model.summary()

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

checkpoint_path = "/content/drive/MyDrive/VCLA-research/cifar10_classifier/training_1/cp-{epoch:04d}.ckpt"
#.format(epoch=EPOCHS)
# checkpoint_dir = os.path.dirname(checkpoint_path)




# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=2*BATCH_SIZE)




with tf.device('/device:GPU:0'):
  model.fit(x_train, y_train, epochs=EPOCHS, validation_data = (x_test, y_test), batch_size=BATCH_SIZE, callbacks=[cp_callback], verbose =1)


loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("final loss: ", loss)
print("final acc: ", accuracy)