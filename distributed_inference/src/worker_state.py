import threading
import select
import socket
import time

class WorkerState:
    def __init__(self, chunk_size) -> None:
        self._chunk_size = chunk_size
        self._next_node = ""
        self._model = ""
        self._weights = ""
        self._lock = threading.Lock()
    @property
    def chunk_size(self):
        with self._lock:
            return self._chunk_size
    @property
    def next_node(self):
        with self._lock:
            return self._next_node
    @next_node.setter
    def next_node(self, nx):
        with self._lock:
            self._next_node = nx
    @property
    def model(self):
        with self._lock:
            return self._model
    @model.setter
    def model(self, m):
        with self._lock:
            self._model = m
    @property
    def weights(self):
        with self._lock:
            return self._weights
    @weights.setter
    def weights(self, w):
        print("Weights set")
        with self._lock:
            self._weights = w
     

    

