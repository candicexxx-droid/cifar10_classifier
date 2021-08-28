import tensorflow as tf
import os
from model import define_compile_model 
import numpy as np 

# if __name__ == "main":

def get_lastest_model(checkpt_path = "training_1/cp-{epoch:04d}.ckpt", data_path="/content/drive/MyDrive/VCLA-research/cifar10_np_data/x_test.npy"):
    
    """
    Return the most updated model from lastest checkpoint and print model's accuracy on test data. 
    data_path: path to .npy test data processed by fetch_data.py(in this repo)
    """
    
    checkpoint_path = checkpt_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print("latest checkpoint: ", latest)
    model = define_compile_model()
    model.load_weights(latest)

    x_test = np.load(data_path)
    y_test = np.load(data_path)

    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Restored model, accuracy(on test data): {:5.2f}%".format(100 * acc))

    return model