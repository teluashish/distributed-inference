import orjson as json
import select
import socket
import threading
from typing import List
import queue
import netron

import numpy as np
import tensorflow as tf

from sock_comm import socket_recv, socket_send

import lz4.frame
import zfpy
import time


class DistributedInference:

    def __init__(self, workerNodeIPs) -> None:
    
        # Replace self.masterIP with machine virtual interface ip from CORE
        self.workerNodeIPs = workerNodeIPs
        self.masterIP = "172.16.0.254" 
        self.chunk_size = 512 * 1000
        
        # Stores Total Overhead
        self.comp_time = 0

    # This function partitions each model into sub-models manually
    def _partition(self, model, layers):
        models = []
        
        input_tensor = model.input
        output_tensor = model.output

        submodel1 = tf.keras.models.Sequential(model.layers[:4])
        output1 = submodel1(input_tensor)
        #submodel1.save("submodel_1.h5")
        #netron.start("submodel_1.h5")

        input2 = tf.keras.layers.Input(shape=output1.shape[1:])
        submodel2 = tf.keras.models.Sequential(model.layers[4:8])
        output2 = submodel2(input2)
        #submodel2.save("submodel_2.h5")
        #netron.start("submodel_2.h5")
     
        input3 = tf.keras.layers.Input(shape=output2.shape[1:])
        submodel3 = tf.keras.models.Sequential(model.layers[8:12])
        output3 = submodel3(input3)
        #submodel3.save("submodel_3.h5")
       #netron.start("submodel_3.h5")
        
        input4 = tf.keras.layers.Input(shape=output3.shape[1:])
        submodel4 = tf.keras.models.Sequential(model.layers[12:16])
        output4 = submodel4(input4)

        input5 = tf.keras.layers.Input(shape=output4.shape[1:])
        submodel5 = tf.keras.models.Sequential(model.layers[16:])
        output5 = submodel5(input5)

        '''
        input6 = tf.keras.layers.Input(shape=output5.shape[1:])
        submodel6 = tf.keras.models.Sequential(model.layers[17:])
        output6 = submodel6(input6)                     
        '''
        models.append(submodel1)
        models.append(submodel2)
        models.append(submodel3)  

        models.append(submodel4)

        models.append(submodel5)

        '''
        models.append(submodel6)                   
        '''
        return models
        
      # This function partitions each model into sub-models specified in the layers argument
    def _partition_by_layers(self, model, layers):
        models = []
       
        for pIdx in range(len(layers)+1):

            start = model.input.name if pIdx == 0 else layers[pIdx-1]
            end = model.output.name if pIdx == len(layers) else layers[pIdx]

            part_name = "part"+str(pIdx+1)
            inpt = tf.keras.Input(tensor=model.get_layer(
                start).output, name=part_name)
            output = self._traverse(model, end, start, part_name, inpt)
            sub_model = tf.keras.Model(inputs=model.get_layer(start).output, outputs=output)
            print(sub_model, ": ", pIdx)
            models.append(sub_model)
  
        return models    
    
        
    # This function traverses and returns a clone of the input graph (model)
    def _traverse(self, model, name, start, part_name,inpt):
        name = name.split('/')[0]
        if name in {start, part_name}:
            return inpt

        output = []
        inbound = model.get_layer(name)._inbound_nodes[0].inbound_layers
        prev_layers = [layer.name for layer in (inbound if type(inbound) == list else [inbound])]

        for prev_layer in prev_layers:
            output.append(self._traverse(
                model, prev_layer, start, part_name, inpt))

        layer = model.get_layer(name)

        return layer(output[0] if len(output) == 1 else output)


    def _create_socket(self, toblock, timeout):
    
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(toblock)
        sock.settimeout(timeout)
        return sock
    
    def _transfer_models(self, models, workerNodeIPs):
    
        for i in range(len(models)):
            
            weights_sock = self._create_socket(0, 10)
            weights_sock.connect((workerNodeIPs[i], 5002))
            
            model_json = models[i].to_json()
            nextWorkerAddress = self.masterIP if (i == len(models) - 1) else workerNodeIPs[i+1]
                
            self._transfer_weights(models[i].get_weights(), weights_sock, self.chunk_size)
            
            model_sock = self._create_socket(0, 10)
            model_sock.connect((workerNodeIPs[i], 5001))
            
            socket_send(model_json.encode(), model_sock, self.chunk_size)
            socket_send(nextWorkerAddress.encode(), model_sock, chunk_size=1)
            
            # Acknowledgement
            select.select([model_sock], [], [])  
            model_sock.recv(1)

    def _transfer_weights(self, weight_data, connection, buffer_size):
        total_size = len(weight_data)
        total_size_in_bytes = total_size.to_bytes(8, 'big')

        while total_size_in_bytes:
            try:
                bytes_sent = connection.send(total_size_in_bytes)
                total_size_in_bytes = total_size_in_bytes[bytes_sent:]
            except socket.error as err:
                if err.errno != socket.EAGAIN:
                    raise err
                select.select([], [connection], [])
                
            for weight_array in weight_data:
                socket_send(self._compressData(weight_array), connection, buffer_size)

    def _compressData(self, arr):
        res=lz4.frame.compress(zfpy.compress_numpy(arr))
            
        return res

    def _decompressData(self, byts):

        res=zfpy.decompress_numpy(lz4.frame.decompress(byts))
   
        return res
        
    def _infer(self, input):
    
        data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_sock.connect((self.workerNodeIPs[0], 5000))
        data_sock.setblocking(0)

        while True:
            model_input = input.get()
            start=time.time()
            out = self._compressData(model_input)
            self.comp_time += (time.time() - start)
            print("overhead so far: ",self.comp_time)
     
            socket_send(out, data_sock, self.chunk_size)

    def _result_server(self, output):
    
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        server.bind(("0.0.0.0", 5000))
        server.listen(1) 
        client = server.accept()[0]
        client.setblocking(0)

        while True:
            data = bytes(socket_recv(client, self.chunk_size))
            start = time.time()
            data_decomp = self._decompressData(data)  
            self.comp_time += (time.time()-start)
            print("overhead so far: ", self.comp_time)
            output.put(data_decomp)

    def start(self, model, partition_layers, ip_stream, op_stream):
    
        models_array = self._partition(model, partition_layers)
        a = threading.Thread(target=self._result_server, args=(op_stream,))
        a.start()
        
        self._transfer_models(models_array, self.workerNodeIPs)
        time.sleep(2)
        
        b = threading.Thread(target=self._infer, args=(ip_stream,), daemon=True)
        b.start()
        a.join()


