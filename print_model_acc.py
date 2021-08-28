#This .py file shows how to retrieve and check accuracy of the trained model 
#Before running this file, make sure you've downloaded the checkpoint folder, processed and save data using fetch_data.py to a directory

from retrieve_model import get_lastest_model

#Be sure to change the following two directories to paths that work on your end
checkpt_path = "training_1/cp-{epoch:04d}.ckpt"
test_data_dir = "/content/drive/MyDrive/VCLA-research/cifar10_np_data/x_test.npy"
model = get_lastest_model(checkpt_path, test_data_dir)