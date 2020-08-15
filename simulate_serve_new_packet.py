# -*- coding:utf-8 -*-
'''
@ author: Orion Han
@ email: jinhan@tju.edu.cn
@ Created on: xxxx-xx-xx
version 1.0
update:
Refer: 

Application: xxxxxx

'''
import socket, time
import numpy as np

s_serve = socket.socket()
s_serve.bind(('127.0.0.1', 4000))
s_serve.listen()
client_socket, client_ip_port = s_serve.accept()

time.sleep(0.04)
# default: int32. 64 channel + 1 label channel, 65*4*40=10400 bytes, and plus 12 bytes header.
send_values = np.random.randint(0, 255, (2603,))

# When connecting successfully, execute the following program.
client_socket.send(send_values)

# a = np.array([19,23,4,254,23])
# client_socket.send(a)

iter_loop = 0
time_buffer = 30  # unit: second
time_packet = 0.04
while True:
    iter_loop += 1
    time.sleep(0.04)
    send_values = np.random.randint(np.mod(iter_loop, 255), np.mod(iter_loop, 255)+1, (2603,))
    client_socket.send(send_values)
    if iter_loop == (time_buffer // time_packet) + 6:
        s_serve.close()
        break

print('breakpoint')
