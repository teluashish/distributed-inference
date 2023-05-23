from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# Specify the path to the saved model file
model_path = '/home/wix/Downloads/nm.h5'

# Load the model
model = load_model(model_path)
import pickle
with open('/home/wix/Downloads/xox', 'rb') as file:
	x = pickle.load(file)
	
y = np.random.randn(*x.shape)

x = np.vstack((x, y))
time_run = 1
in_sec = 100
start = time.time()
result_count = 0
i=0
while (time.time() - start) < in_sec:
    model.predict(x[i:i+2,:,:])

    print(i)
    print(time.time() - start)
    i+=2
    result_count += 1
print(f"In {time_run} min, {result_count} results")
