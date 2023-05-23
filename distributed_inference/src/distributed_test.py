from master import DistributedInference
import threading
import tensorflow as tf
import keras.applications
from tensorflow.keras.models import load_model
import numpy as np
import queue
import time
import pickle

workerNodeIPs = ["172.16.0.2","172.16.0.3","172.16.0.4","172.16.0.5","172.16.0.6"]#,"172.16.0.7"]
distributedInference = DistributedInference(workerNodeIPs)

# Specify the path to the saved model file
model_path = '/home/wix/Downloads/nm.h5'
part_at=0
# Load the model
model = load_model(model_path)
with open('/home/wix/Downloads/xox', 'rb') as file:
	x = pickle.load(file)


time_min = 1
run = 100


def print_result(q):
    res_count = 0
    start = time.time()
    while (time.time() - start) < run:

        #print(q.qsize())
        res = q.get()
        #print(res)

        res_count += 1

        if res_count==1:
            print(res_count)
            toti = time.time()
        
        #print(res.shape)
        #print(time.time() - start)
    ttt=time.time()-toti
    print(f"{res_count} results in {ttt} seconds")
    print(f"Throughput: {res_count / ttt}")
    exit(0)

input_q = queue.Queue(10)
output_q = queue.Queue(10)

a = threading.Thread(target = distributedInference.start, args=(model, part_at, input_q, output_q), daemon=True)
b = threading.Thread(target=print_result, args=(output_q,))
a.start()
b.start()

for i in range(240):
    # Whatever input you want
    input_q.put(x[i:i+2,:,:])

b.join()
