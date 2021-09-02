# cifar10_classifier

Reference: 
https://www.kaggle.com/kutaykutlu/resnet50-transfer-learning-cifar-10-beginner

link to checkpoint folder: 
https://drive.google.com/drive/folders/1Xa61awYeoYnQKJ6PD_JNTO_nAOmOnM7X?usp=sharing

Transfer learning with ResNet50 on cifar10 image data

Restored model, accuracy(on test data): 94.50%



To use the trained model, check out print_model_acc.py 

Before running print_model_acc.py, make sure you've downloaded the checkpoint folder, processed and save data using fetch_data.py to a directory


Notes: 

fetch_data.py-download,split and preprocess cifar10 data 

model.py - define model architecture

train.py - model training and save checkpoint files 

retrieve_model.py - a function for retrieving a trained model

print_model_acc.py - show how to implement the retrieve model function from retrieve_model.py

