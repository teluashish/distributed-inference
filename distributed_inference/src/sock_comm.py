import select
import socket

def socket_send(bytes, sock, chunk_size):

    size = len(bytes)
    size_bytes = size.to_bytes(8, 'big')
    
    while len(size_bytes) > 0:
        try:
            size_bytes = size_bytes[sock.send(size_bytes):]
        except socket.error as e:
            if e.errno != socket.EAGAIN:
                raise e
            select.select([], [sock], [])
                
    for i in range(0, len(bytes), chunk_size):
        chunk = bytes[i:] if len(bytes) - i < chunk_size else bytes[i:i+chunk_size]
        while len(chunk) > 0:
            try:
                chunk = chunk[sock.send(chunk):]
            except socket.error as e:
                if e.errno != socket.EAGAIN:
                    raise e
                select.select([], [sock], [])
               
def socket_recv(sock: socket.socket, chunk_size: int):
   
    size_left = 8
    byts = bytearray()
    while size_left > 0:
        try: 
            recv = sock.recv(min(size_left, 8))
            size_left -= len(recv)
            byts.extend(recv)
        except socket.error as e:
            if e.errno != socket.EAGAIN:
                raise e
            select.select([sock], [], [])
            
    left = data_size = int.from_bytes(byts, 'big')
    data_json = bytearray(data_size)
    data_counter = 0
    
    while (left > 0):
        try:
            recv = sock.recv(min(left, chunk_size))
            left -= len(recv)
            data_json[data_counter:data_counter+len(recv)] = recv
            data_counter += len(recv)
            
        except socket.error as e:
            if e.errno != socket.EAGAIN:
                raise e
            select.select([sock], [], [])
    return data_json
